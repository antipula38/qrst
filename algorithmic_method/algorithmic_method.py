# Скачивания и импорты
!pip install mne
!pip install wfdb
!pip install PyQt5
!pip install tqdm
import numpy as np
from scipy.ndimage import gaussian_filter1d
import mne
import matplotlib.pyplot as plt
import wfdb
import os
from pathlib import Path
import PyQt5
from scipy import signal
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import pandas as pd
from scipy.signal import find_peaks
import pywt


# Написанный детектор
class PQRSTDetector:

    def __init__(self, fs=500):
        self.fs = fs

        # Параметры для билатерального фильтра
        self.sigma_d_max = 5  # максимальное стандартное отклонение для пространственного фильтра
        self.sigma_r_max = 0.5  # максимальное стандартное отклонение для диапазонного фильтра
        self.sigma_min = 1e-5  # минимальная сигма (чтобы избежать деления на 0)
        self.sigma2_max = 0.131  # максимальная дисперсия

        self.window_size = max(10, int(15 * fs / 370))  # размер окна билатерального фильтра
        if self.window_size % 2 == 0:  # делаем нечетным для симметрии
            self.window_size += 1

        # Длительности интервалов (верхние границы нормы)
        self.qrs_duration = 0.1  # длительность QRS-комплекса
        self.pr_interval = 0.2  # PR интервал
        self.qt_interval = 0.44  # QT интервал

        # Параметры для Pan-Tompkins
        self.lowcut = 1.5  # полоса пропускания для QRS
        self.highcut = 45.0
        self._create_pan_tompkins_filters()  # создаем фильтр

        # Параметры для поиска границ
        self.baseline_threshold = 0.015  # порог отклонения от baseline для границ
        self.slope_threshold = 0.002  # порог для производной (чувствительность к изменению наклона)

    # Предобработка сигнала
    def preprocess_signal(self, ecg_signal):
        ecg_detrend = signal.detrend(ecg_signal)  # убираем линейный тренд

        # полосовой фильтр
        nyquist = self.fs / 2
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = signal.butter(3, [low, high], btype='band')
        ecg_filtered = signal.filtfilt(b, a, ecg_detrend)
        return ecg_filtered

    # Адаптивный билатеральный фильтр
    def adaptive_bilateral_filter(self, signal):
        n = len(signal)
        filtered = np.zeros_like(signal)
        half = self.window_size // 2

        # Уменьшаем максимальные sigma для более мягкой фильтрации
        sigma_d_max = 5
        sigma_r_max = 0.5

        for i in range(n):
            start = max(0, i - half)
            end = min(n, i + half + 1)
            window = signal[start:end]
            variance = np.var(window)

            # Модифицированная адаптация
            sigma_d = np.log2((variance / self.sigma2_max + 1) ** sigma_d_max) + self.sigma_min
            sigma_r = np.log2((variance / self.sigma2_max + 1) ** sigma_r_max) + self.sigma_min

            sigma_d = min(max(sigma_d, self.sigma_min), sigma_d_max)
            sigma_r = min(max(sigma_r, self.sigma_min), sigma_r_max)

            spatial_weights = np.exp(-((np.arange(start, end) - i) ** 2) / (2 * sigma_d ** 2))
            range_weights = np.exp(-((window - signal[i]) ** 2) / (2 * sigma_r ** 2))
            weights = spatial_weights * range_weights

            if weights.sum() > 0:
                filtered[i] = np.sum(window * weights) / weights.sum()
            else:
                filtered[i] = signal[i]

        return filtered

    # Создание фильтров Pan-Tompkins
    def _create_pan_tompkins_filters(self):
        nyquist = 0.5 * self.fs  # Частота Найквиста (половина частоты дискретизации)

        # Нормированные частоты для фильтров
        low = self.lowcut / nyquist
        high = self.highcut / nyquist

        self.bp_b, self.bp_a = signal.butter(2, [low, high], btype='band')  # фильтр Баттерворта 2го порядка
        self.deriv_coeffs = np.array([-1, -2, 0, 2, 1]) / (
                    8 * self.fs)  # коэффициенты для вычисления производной из статьи по Пану-Томпкинсу

    # QRS-detector по Пану-Томпкинсу
    def pan_tompkins_qrs_detector(self, ecg_signal):
        filtered = signal.filtfilt(self.bp_b, self.bp_a, ecg_signal)  # фильтрация без сдвига сигнала
        derivative = np.convolve(filtered, self.deriv_coeffs, mode='same')  # свертка для вычисления производной
        squared = derivative ** 2  # усиливаем QRS возведением в квадрат
        window = int(0.15 * self.fs)  # скользящее окно
        integrated = np.convolve(squared, np.ones(window) / window, mode='same')  # интеграция со скользящим окном

        # Адаптивный порог
        signal_peak = np.max(integrated) * 0.6  # Сигнал
        noise_peak = np.mean(integrated) * 0.5  # Шум
        threshold = noise_peak + 0.25 * (signal_peak - noise_peak)

        candidates = integrated > threshold  # кондидаты в R-пики
        r_peaks = []

        # В каждом интервале-кандидате ищем максимум
        i = 0
        while i < len(candidates):
            if candidates[i]:
                start = i
                while i < len(candidates) and candidates[i]:
                    i += 1
                end = i

                # Ищем максимум на оригинальном сигнале
                region = ecg_signal[start:end]
                if len(region) > 0:
                    r_idx = start + np.argmax(region)
                    r_peaks.append(r_idx)
            else:
                i += 1

        # Рефрактерный период 200 мс - удаляем слишком близкие пики
        min_distance = int(0.2 * self.fs)
        filtered_peaks = []
        for peak in r_peaks:
            if not filtered_peaks or peak - filtered_peaks[-1] > min_distance:
                filtered_peaks.append(peak)

        return np.array(filtered_peaks)

    # Добавляем к детекции R-пиков типизацию
    def detect_r_peaks(self, ecg_signal):
        # Pan-Tompkins
        r_peaks_pt = self.pan_tompkins_qrs_detector(ecg_signal)

        # Проверка границ
        r_peaks = []
        for r_idx in r_peaks_pt:
            if r_idx < 2 or r_idx > len(ecg_signal) - 3:
                r_peaks.append(r_idx)
                continue

            # Проверка влево-вправо, что это реально максимумы
            local_max = True
            for j in range(1, 4):
                if r_idx - j >= 0:
                    if ecg_signal[r_idx] <= ecg_signal[r_idx - j]:
                        local_max = False
                        break
                if r_idx + j < len(ecg_signal):
                    if ecg_signal[r_idx] <= ecg_signal[r_idx + j]:
                        local_max = False
                        break

            if local_max:
                r_peaks.append(r_idx)

        # Если после проверки осталось мало пиков, возвращаем исходные
        if len(r_peaks) < len(r_peaks_pt) * 0.7:
            r_peaks = r_peaks_pt.tolist()

        # Типизация
        I = (ecg_signal - np.min(ecg_signal)) / (np.max(ecg_signal) - np.min(ecg_signal))  # Нормализация сигнала
        B = self.adaptive_bilateral_filter(I)  # Предсказание фона

        # Определяем выше или ниже R-пик в сравнении с baseline
        r_types = []
        for r_idx in r_peaks:
            if r_idx < 5 or r_idx > len(I) - 5:
                r_types.append(1)
                continue
            local_region = I[r_idx - 5:r_idx + 5]
            r_types.append(1 if np.mean(local_region) > 0.5 else -1)

        return np.array(r_peaks), np.array(r_types), B, I - B, I

    # находим точки Q
    def detect_q_points(self, ecg_signal, r_peaks, r_types):
        I = (ecg_signal - np.min(ecg_signal)) / (np.max(ecg_signal) - np.min(ecg_signal))  # нормализация
        q_points = []

        # ищем слева от R
        for r_idx, r_type in zip(r_peaks, r_types):
            search_start = max(0, r_idx - int(0.1 * self.fs))
            search_end = r_idx - 3

            if search_end <= search_start:
                q_points.append(r_idx - int(0.03 * self.fs))
                continue

            # Для положительных пиков ищем локальный минимум
            region = I[search_start:search_end]
            if len(region) > 0:
                if r_type == 1:
                    valleys, _ = find_peaks(-region, distance=5)
                    if len(valleys) > 0:
                        best_valley = valleys[np.argmin(np.abs(valleys - (r_idx - search_start)))]
                        q_points.append(search_start + best_valley)
                    else:
                        q_points.append(search_start + np.argmin(region))
                else:
                    peaks, _ = find_peaks(region, distance=5)
                    if len(peaks) > 0:
                        best_peak = peaks[np.argmin(np.abs(peaks - (r_idx - search_start)))]
                        q_points.append(search_start + best_peak)
                    else:
                        q_points.append(search_start + np.argmax(region))
            else:
                q_points.append(r_idx - int(0.03 * self.fs))

        return np.array(q_points)

    # Ищем S точки уже справа от пиков
    def detect_s_points(self, ecg_signal, r_peaks, r_types):
        I = (ecg_signal - np.min(ecg_signal)) / (np.max(ecg_signal) - np.min(ecg_signal))
        s_points = []

        for r_idx, r_type in zip(r_peaks, r_types):
            search_start = r_idx + 3
            search_end = min(len(I), r_idx + int(0.1 * self.fs))

            if search_end <= search_start:
                s_points.append(r_idx + int(0.03 * self.fs))
                continue

            region = I[search_start:search_end]
            if len(region) > 0:
                if r_type == 1:
                    valleys, _ = find_peaks(-region, distance=5)
                    if len(valleys) > 0:
                        best_valley = valleys[np.argmin(np.abs(valleys))]
                        s_points.append(search_start + best_valley)
                    else:
                        s_points.append(search_start + np.argmin(region))
                else:
                    peaks, _ = find_peaks(region, distance=5)
                    if len(peaks) > 0:
                        best_peak = peaks[np.argmin(np.abs(peaks))]
                        s_points.append(search_start + best_peak)
                    else:
                        s_points.append(search_start + np.argmax(region))
            else:
                s_points.append(r_idx + int(0.03 * self.fs))

        return np.array(s_points)

    # Ищем Р точки
    def detect_p_points(self, background_signal, q_points):
        B = background_signal
        p_points = []
        p_types = []

        for q_idx in q_points:
            search_start = max(0, q_idx - int(0.25 * self.fs))
            search_end = q_idx - int(0.05 * self.fs)

            if search_end <= search_start:
                continue

            region = B[search_start:search_end]
            if len(region) < 15:
                continue

            peaks, peak_props = find_peaks(region, distance=8, prominence=0.01)  # prominence подбирала
            valleys, valley_props = find_peaks(-region, distance=8, prominence=0.01)

            candidates = []

            for p, prom in zip(peaks, peak_props['prominences']):
                if prom > 0.01:
                    candidates.append((search_start + p, prom, 1, region[p]))

            for v, prom in zip(valleys, valley_props['prominences']):
                if prom > 0.01:
                    candidates.append((search_start + v, prom, -1, region[v]))

            if not candidates:
                continue

            best_idx, best_prom, best_type, best_val = max(candidates, key=lambda x: x[1])

            mean_val = np.mean(region)
            amplitude = abs(best_val - mean_val)

            if amplitude < 0.01:
                continue

            p_points.append(best_idx)
            p_types.append(best_type)

        return np.array(p_points), np.array(p_types)

    # Аналогично ищем Т справа от S
    def detect_t_points(self, background_signal, s_points):
        B = background_signal
        t_points = []
        t_types = []

        for s_idx in s_points:
            search_start = s_idx + int(0.1 * self.fs)
            search_end = min(len(B), s_idx + int(0.42 * self.fs))

            if search_end <= search_start:
                continue

            region = B[search_start:search_end]
            if len(region) < 20:
                continue

            peaks, peak_props = find_peaks(region, distance=12, prominence=0.02)  # prominence подбирала
            valleys, valley_props = find_peaks(-region, distance=12, prominence=0.02)

            candidates = []

            for p, prom in zip(peaks, peak_props['prominences']):
                if prom > 0.02:
                    candidates.append((search_start + p, prom, 1, region[p]))

            for v, prom in zip(valleys, valley_props['prominences']):
                if prom > 0.02:
                    candidates.append((search_start + v, prom, -1, region[v]))

            if not candidates:
                continue

            best_idx, best_prom, best_type, best_val = max(candidates, key=lambda x: x[1])

            mean_val = np.mean(region)
            if abs(best_val - mean_val) < 0.015:
                continue

            t_points.append(best_idx)
            t_types.append(best_type)

        return np.array(t_points), np.array(t_types)

    def remove_pt_duplicates(self, p_points, p_types, t_points, t_types):
        if len(p_points) == 0 or len(t_points) == 0:
            return p_points, p_types, t_points, t_types

        # Создаем список всех P и T точек
        all_pt = []
        for p in p_points:
            all_pt.append((p, 'P'))
        for t in t_points:
            all_pt.append((t, 'T'))

        # Сортируем по времени
        all_pt.sort(key=lambda x: x[0])

        # Удаляем дубликаты (точки, которые слишком близко)
        unique_pt = []
        for i, (idx, ptype) in enumerate(all_pt):
            is_duplicate = False
            for prev_idx, prev_ptype in unique_pt:
                if abs(idx - prev_idx) < int(0.1 * fs):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_pt.append((idx, ptype))

        # Разделяем обратно
        new_p_points = []
        new_p_types = []
        new_t_points = []
        new_t_types = []

        # Создаем словари для быстрого поиска типов
        p_dict = {p: pt for p, pt in zip(p_points, p_types)}
        t_dict = {t: tt for t, tt in zip(t_points, t_types)}

        for idx, ptype in unique_pt:
            if ptype == 'P':
                new_p_points.append(idx)
                new_p_types.append(p_dict[idx])
            else:
                new_t_points.append(idx)
                new_t_types.append(t_dict[idx])

        return np.array(new_p_points), np.array(new_p_types), np.array(new_t_points), np.array(new_t_types)

    # Ищем границы волн
    def find_wave_boundary(self, signal, peak_idx, peak_type, direction='left', max_search=0.15):
        max_offset = int(max_search * self.fs)  # время поиска -> отсчеты

        # Поиск слева от пика
        if direction == 'left':
            search_start = max(0, peak_idx - max_offset)
            search_end = peak_idx - 3

            if search_end <= search_start:
                return max(0, peak_idx - int(0.05 * self.fs))

            deriv = np.diff(signal)
            baseline = np.mean(signal[search_start:search_start + int(0.03 * self.fs)])

            for i in range(search_end, search_start, -1):
                if i <= 0 or i >= len(signal) - 1:
                    continue

                if peak_type == 1:
                    if signal[i] <= baseline + self.baseline_threshold and abs(deriv[i - 1]) < self.slope_threshold:
                        return i
                else:
                    if signal[i] >= baseline - self.baseline_threshold and abs(deriv[i - 1]) < self.slope_threshold:
                        return i

            return max(0, peak_idx - int(0.05 * self.fs))

        # Поиск справа от пика
        else:
            search_start = peak_idx + 3
            search_end = min(len(signal), peak_idx + max_offset)

            if search_end <= search_start:
                return min(len(signal) - 1, peak_idx + int(0.05 * self.fs))

            deriv = np.diff(signal)
            baseline_start = min(len(signal) - 1, search_end - int(0.05 * self.fs))
            baseline = np.mean(signal[baseline_start:search_end])

            for i in range(search_start, search_end):
                if i >= len(signal) - 1:
                    continue

                if peak_type == 1:
                    if abs(signal[i] - baseline) < self.baseline_threshold and abs(deriv[i]) < self.slope_threshold:
                        return i
                else:
                    if abs(signal[i] - baseline) < self.baseline_threshold and abs(deriv[i]) < self.slope_threshold:
                        return i

            return min(len(signal) - 1, peak_idx + int(0.05 * self.fs))

    def detect_p_onset_offset(self, background_signal, p_points, p_types):
        p_onsets = []
        p_offsets = []

        for p_idx, p_type in zip(p_points, p_types):
            onset = self.find_wave_boundary(background_signal, p_idx, p_type, 'left', 0.1)
            offset = self.find_wave_boundary(background_signal, p_idx, p_type, 'right', 0.12)

            if onset >= p_idx:
                onset = p_idx - int(0.05 * self.fs)
            if offset <= p_idx:
                offset = p_idx + int(0.05 * self.fs)

            p_onsets.append(onset)
            p_offsets.append(offset)

        return np.array(p_onsets), np.array(p_offsets)

    def detect_t_onset_offset(self, background_signal, t_points, t_types):
        t_onsets = []
        t_offsets = []

        for t_idx, t_type in zip(t_points, t_types):
            onset = self.find_wave_boundary(background_signal, t_idx, t_type, 'left', 0.15)
            offset = self.find_wave_boundary(background_signal, t_idx, t_type, 'right', 0.15)

            if onset >= t_idx:
                onset = t_idx - int(0.07 * self.fs)
            if offset <= t_idx:
                offset = t_idx + int(0.07 * self.fs)

            t_onsets.append(onset)
            t_offsets.append(offset)

        return np.array(t_onsets), np.array(t_offsets)

    def detect_qrs_onset_offset(self, ecg_signal, r_peaks, r_types):
        q_points = self.detect_q_points(ecg_signal, r_peaks, r_types)
        qrs_onsets = []

        for q_idx, r_type in zip(q_points, r_types):
            onset = self.find_wave_boundary(ecg_signal, q_idx, r_type, 'left', 0.05)
            qrs_onsets.append(onset)

        s_points = self.detect_s_points(ecg_signal, r_peaks, r_types)
        qrs_offsets = []

        for s_idx, r_type in zip(s_points, r_types):
            offset = self.find_wave_boundary(ecg_signal, s_idx, r_type, 'right', 0.05)
            qrs_offsets.append(offset)

        return np.array(qrs_onsets), np.array(qrs_offsets)

    def process(self, ecg_signal):
        ecg_clean = self.preprocess_signal(ecg_signal)

        r_peaks, r_types, background, diff_signal, normalized = self.detect_r_peaks(ecg_clean)

        q_points = self.detect_q_points(ecg_clean, r_peaks, r_types)
        s_points = self.detect_s_points(ecg_clean, r_peaks, r_types)
        p_points, p_types = self.detect_p_points(background, q_points)
        t_points, t_types = self.detect_t_points(background, s_points)

        p_points, p_types, t_points, t_types = self.remove_pt_duplicates(p_points, p_types, t_points, t_types)

        p_onsets, p_offsets = self.detect_p_onset_offset(background, p_points, p_types)
        t_onsets, t_offsets = self.detect_t_onset_offset(background, t_points, t_types)
        qrs_onsets, qrs_offsets = self.detect_qrs_onset_offset(ecg_clean, r_peaks, r_types)

        onsets = np.sort(np.concatenate([p_onsets, t_onsets, qrs_onsets]))
        offsets = np.sort(np.concatenate([p_offsets, t_offsets, qrs_offsets]))

        results = {
            'r_peaks': r_peaks,
            'r_types': r_types,
            'q_points': q_points,
            's_points': s_points,
            'p_points': p_points,
            'p_types': p_types,
            't_points': t_points,
            't_types': t_types,
            'onsets': onsets,
            'offsets': offsets,
            'background': background,
            'difference': diff_signal,
            'normalized': normalized
        }

        return results

    def plot_results(self, ecg_signal, results, tmin=0, tmax=10):
        times = np.arange(len(ecg_signal)) / self.fs
        mask = (times >= tmin) & (times <= tmax)
        data_seg = ecg_signal[mask]
        times_seg = times[mask]

        plt.figure(figsize=(15, 6))
        plt.plot(times_seg, data_seg, 'b-', linewidth=1, alpha=0.7, label='ЭКГ сигнал')

        plot_config = [
            ('r_peaks', 'red', '^', 'R-пики', 120, 'R'),
            ('q_points', 'orange', 's', 'Q-точки', 100, 'Q'),
            ('s_points', 'purple', 'D', 'S-точки', 100, 'S'),
            ('p_points', 'green', 'o', 'P-точки', 100, 'P'),
            ('t_points', 'brown', 'v', 'T-точки', 100, 'T'),
            ('onsets', 'pink', '<', 'Начала волн (', 80, '('),
            ('offsets', 'lightblue', '>', 'Концы волн )', 80, ')')
        ]

        legend_added = set()

        for key, color, marker, label_name, size, text in plot_config:
            if key in results and len(results[key]) > 0:
                points = results[key]
                for idx in points:
                    t = idx / self.fs
                    if tmin <= t <= tmax:
                        seg_idx = np.argmin(np.abs(times_seg - t))
                        y_val = data_seg[seg_idx]

                        if label_name not in legend_added:
                            plt.scatter(t, y_val, color=color, marker=marker, s=size,
                                        edgecolors='black', linewidth=2, zorder=5,
                                        label=label_name)
                            legend_added.add(label_name)
                        else:
                            plt.scatter(t, y_val, color=color, marker=marker, s=size,
                                        edgecolors='black', linewidth=2, zorder=5)

                        if key in ['r_peaks', 'q_points', 's_points', 'p_points', 't_points']:
                            plt.annotate(text, (t, y_val),
                                         xytext=(5, 5), textcoords='offset points',
                                         fontsize=9, color='black', fontweight='bold')

        plt.xlabel('Время (с)', fontsize=12)
        plt.ylabel('Амплитуда (мВ)', fontsize=12)
        plt.title(f'Детекция PQRST с границами волн ({tmin}-{tmax} с)')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', ncol=3, fontsize=10)
        plt.tight_layout()
        plt.show()

# База данных LUDB
!wget -r -N -c -np https://physionet.org/files/ludb/1.0.1/
data_path = 'physionet.org/files/ludb/1.0.1/data/'
record_name = "145"
chanels_list = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
channel = chanels_list[0]
# Полный путь к файлу без расширения
record_path = os.path.join(data_path, record_name)
record = wfdb.rdrecord(record_path)
sfreq = record.fs  # Частота дискретизации
channels = record.sig_name  # Названия каналов
data = record.p_signal.T  # Транспонируем для MNE (каналы x отсчеты)
# Создание информации для MNE
ch_types = ['ecg'] * len(channels)
info = mne.create_info(
    ch_names=channels,
    ch_types=ch_types,
    sfreq=sfreq
)
raw = mne.io.RawArray(data, info)


# Графики для LUDB
def plot_ecg_with_annotations(raw, channel, tmin=0, tmax=10):
    """
    Построение графика ЭКГ с аннотациями сокращений для указанного канала

    Parameters:
    -----------
    raw : mne.io.Raw
        Объект Raw с данными
    channel : str
        Конкретный канал для отображения ('i', 'ii', 'iii', и т.д.)
    tmin : float
        Начальное время в секундах
    tmax : float
        Конечное время в секундах
    """

    # Проверка наличия канала
    if channel not in raw.ch_names:
        print(f"Канал '{channel}' не найден. Доступные каналы: {raw.ch_names}")
        return None, None

    # Получение данных для указанного временного интервала
    data, times = raw.get_data(
        picks=channel,
        tmin=tmin,
        tmax=tmax,
        return_times=True
    )
    annotation = wfdb.rdann(record_path, channel)
    # Получение аннотаций
    annot_times = annotation.sample / sfreq

    # Символы аннотаций (типы сокращений)
    annot_labels = annotation.symbol

    # Описание аннотаций (длительность - 0 для точечных аннотаций)
    annot_durations = [0] * len(annot_times)

    # Создание аннотаций MNE
    annotations = mne.Annotations(
        onset=annot_times,
        duration=annot_durations,
        description=annot_labels
    )

    # Создание графика
    fig, ax = plt.subplots(figsize=(15, 6))

    # Построение сигнала ЭКГ
    ax.plot(times, data[0], 'b-', linewidth=1, label=f'ЭКГ {channel.upper()}')

    # Добавление меток сокращений
    if len(annotations) > 0:
        # Фильтрация аннотаций в выбранном временном окне
        mask = (annotations.onset >= tmin) & (annotations.onset <= tmax)
        annot_times_win = annotations.onset[mask]
        annot_labels_win = np.array(annotations.description)[mask]

        # Поиск соответствующих значений сигнала для каждой аннотации
        for i, (t, label) in enumerate(zip(annot_times_win, annot_labels_win)):
            # Находим ближайший индекс к времени аннотации
            idx = np.argmin(np.abs(times - t))
            y_val = data[0, idx]

            # Цвет в зависимости от типа сокращения
            if label == 'N':
                color = 'green'
                marker = 'o'
                label_text = 'Нормальное'
            elif label == 'V':
                color = 'red'
                marker = '^'
                label_text = 'Желудочковая экстрасистола'
            elif label == 'A':
                color = 'orange'
                marker = 's'
                label_text = 'Предсердная экстрасистола'
            elif label == 'R':
                color = 'cyan'
                marker = 'd'
                label_text = 'Блокада правой ножки'
            elif label == 'L':
                color = 'magenta'
                marker = 'p'
                label_text = 'Блокада левой ножки'
            elif label == 'E':
                color = 'brown'
                marker = 'h'
                label_text = 'Желудочковый убегающий'
            else:
                color = 'purple'
                marker = 'D'
                label_text = f'{label}'

            # Отображаем только первую метку в легенде для каждого типа
            if i == 0 or label not in [l for l in annot_labels_win[:i]]:
                ax.plot(t, y_val, marker=marker, color=color, markersize=8,
                        label=f'{label} - {label_text}')
            else:
                ax.plot(t, y_val, marker=marker, color=color, markersize=8)

            # Добавление текстовой метки
            ax.annotate(label, (t, y_val),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color=color, fontweight='bold')

    # Настройка графика
    ax.set_xlabel('Время (с)', fontsize=12)
    ax.set_ylabel('Амплитуда (мВ)', fontsize=12)
    ax.set_title(f'ЭКГ - канал {channel.upper()} с метками сердечных сокращений', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.show()

    return fig, ax


record_name = "14"
channel = 'i'

record_path = os.path.join(data_path, record_name)
record = wfdb.rdrecord(record_path)
fs = record.fs

annotation = wfdb.rdann(record_path, channel)

ch_types = ['ecg'] * len(record.sig_name)
info = mne.create_info(
    ch_names=record.sig_name,
    ch_types=ch_types,
    sfreq=fs
)

annot_times = annotation.sample / fs
annot_durations = [0] * len(annot_times)
mne_annotations = mne.Annotations(
    onset=annot_times,
    duration=annot_durations,
    description=annotation.symbol
)

raw = mne.io.RawArray(record.p_signal.T, info)
raw.set_annotations(mne_annotations)

data_for_detector, _ = raw.get_data(picks=channel, tmin=0, tmax=10, return_times=True)
ecg_segment = data_for_detector[0]

detector = PQRSTDetector(fs=fs)

results = detector.process(ecg_segment)

detector.plot_results(ecg_segment, results, tmin=0, tmax=10)

fig, ax = plot_ecg_with_annotations(raw, channel, tmin=0, tmax=10)


# Метрики для LUDB
def evaluate_on_all(data_path, detector_class, channel='ii', window=0.25, max_records=None,
                    skip_detector_points={'r_peaks': 0, 'p_points': 0, 't_points': 0, 'onsets': 0, 'offsets': 0}):
    all_files = os.listdir(data_path)
    record_names = sorted(list(set([f.split('.')[0] for f in all_files if f.endswith('.dat')])))

    if max_records:
        record_names = record_names[:max_records]

    mapping = {
        'r_peaks': (['R', 'N'], 'R-пики'),
        'p_points': (['p'], 'P-точки'),
        't_points': (['t'], 'T-точки'),
        'onsets': (['('], 'Начала волн (P+T+QRS)'),
        'offsets': ([')'], 'Концы волн (P+T+QRS)')
    }

    all_results = []
    total_records_processed = 0
    records_with_channel = 0
    records_without_channel = 0

    total_stats = {
        'tp': {key: 0 for key in mapping.keys()},
        'fp': {key: 0 for key in mapping.keys()},
        'fn': {key: 0 for key in mapping.keys()},
        'errors': {key: [] for key in mapping.keys()}
    }
    total_stats['tp']['total'] = 0
    total_stats['fp']['total'] = 0
    total_stats['fn']['total'] = 0
    total_stats['errors']['total'] = []

    for record_name in tqdm(record_names, desc="Обработка записей"):
        total_records_processed += 1
        try:
            record_path = os.path.join(data_path, record_name)
            record = wfdb.rdrecord(record_path)
            fs = record.fs

            if channel not in record.sig_name:
                records_without_channel += 1
                continue

            records_with_channel += 1
            channel_idx = record.sig_name.index(channel)
            ecg_data = record.p_signal[:, channel_idx]

            try:
                annotation = wfdb.rdann(record_path, channel)
            except:
                continue

            annot_times = annotation.sample / fs
            annot_labels = annotation.symbol

            ludb_annotations = {}
            for time, label in zip(annot_times, annot_labels):
                if label not in ludb_annotations:
                    ludb_annotations[label] = []
                ludb_annotations[label].append(time)

            for label in ludb_annotations:
                ludb_annotations[label] = np.sort(ludb_annotations[label])

            detector = detector_class(fs=fs)
            detector_results = detector.process(ecg_data)

            det_points = {}
            for key in mapping.keys():
                if key in detector_results and len(detector_results[key]) > 0:
                    points = np.sort(detector_results[key] / fs)
                    skip = skip_detector_points.get(key, 0)
                    if skip > 0 and len(points) > skip:
                        points = points[skip:]
                    det_points[key] = points
                else:
                    det_points[key] = np.array([])

            record_metrics = {'record': record_name}

            for det_key, (gt_symbols, name) in mapping.items():
                gt_points = []
                for sym in gt_symbols:
                    if sym in ludb_annotations:
                        gt_points.extend(ludb_annotations[sym])
                gt_points = np.sort(gt_points)

                det = det_points[det_key]

                if len(det) == 0 and len(gt_points) == 0:
                    tp = fp = fn = 0
                    errors = []
                elif len(det) == 0:
                    tp = 0
                    fp = 0
                    fn = len(gt_points)
                    errors = []
                elif len(gt_points) == 0:
                    tp = 0
                    fp = len(det)
                    fn = 0
                    errors = []
                else:
                    matched_det = np.zeros(len(det), dtype=bool)
                    matched_gt = np.zeros(len(gt_points), dtype=bool)
                    errors = []

                    for i, gt in enumerate(gt_points):
                        if not matched_gt[i]:
                            distances = np.abs(det - gt)
                            distances[matched_det] = np.inf

                            if len(distances) > 0:
                                min_idx = np.argmin(distances)
                                if distances[min_idx] <= window:
                                    matched_det[min_idx] = True
                                    matched_gt[i] = True
                                    error = (det[min_idx] - gt) * 1000
                                    errors.append(error)

                    tp = len(errors)
                    fp = len(det) - tp
                    fn = len(gt_points) - tp

                precision = tp / (tp + fp) * 100 if tp + fp > 0 else 0
                recall = tp / (tp + fn) * 100 if tp + fn > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

                mean_error = np.mean(errors) if errors else None

                record_metrics[f'{det_key}_tp'] = tp
                record_metrics[f'{det_key}_fp'] = fp
                record_metrics[f'{det_key}_fn'] = fn
                record_metrics[f'{det_key}_precision'] = precision
                record_metrics[f'{det_key}_recall'] = recall
                record_metrics[f'{det_key}_f1'] = f1
                record_metrics[f'{det_key}_mean_error'] = mean_error
                record_metrics[f'{det_key}_detected'] = len(det)
                record_metrics[f'{det_key}_ground_truth'] = len(gt_points)

                total_stats['tp'][det_key] += tp
                total_stats['fp'][det_key] += fp
                total_stats['fn'][det_key] += fn
                total_stats['errors'][det_key].extend(errors)
                total_stats['tp']['total'] += tp
                total_stats['fp']['total'] += fp
                total_stats['fn']['total'] += fn
                total_stats['errors']['total'].extend(errors)

            all_results.append(record_metrics)

        except Exception as e:
            print(f"\nОшибка при обработке {record_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if all_results:
        df_records = pd.DataFrame(all_results)
    else:
        df_records = pd.DataFrame()
        return df_records, pd.DataFrame()

    print(f"Обработано с каналом {channel}: {records_with_channel}")

    summary = {}

    for det_key, (_, name) in mapping.items():
        tp = total_stats['tp'][det_key]
        fp = total_stats['fp'][det_key]
        fn = total_stats['fn'][det_key]
        errors = total_stats['errors'][det_key]

        precision = tp / (tp + fp) * 100 if tp + fp > 0 else 0
        recall = tp / (tp + fn) * 100 if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        mean_error = np.mean(errors) if errors else None

        summary[det_key] = {
            'Тип': name,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'Precision %': round(precision, 2),
            'Recall %': round(recall, 2),
            'F1 %': round(f1, 2),
            'Ср.ошибка мс': round(mean_error, 2) if mean_error else None
        }

    total_tp = total_stats['tp']['total']
    total_fp = total_stats['fp']['total']
    total_fn = total_stats['fn']['total']
    total_errors = total_stats['errors']['total']

    total_precision = total_tp / (total_tp + total_fp) * 100 if total_tp + total_fp > 0 else 0
    total_recall = total_tp / (total_tp + total_fn) * 100 if total_tp + total_fn > 0 else 0
    total_f1 = 2 * total_precision * total_recall / (
                total_precision + total_recall) if total_precision + total_recall > 0 else 0

    summary['total'] = {
        'Тип': f'ВСЕГО (канал {channel})',
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn,
        'Precision %': round(total_precision, 2),
        'Recall %': round(total_recall, 2),
        'F1 %': round(total_f1, 2),
        'Ср.ошибка мс': round(np.mean(total_errors), 2) if total_errors else None
    }

    df_summary = pd.DataFrame(summary).T

    return df_records, df_summary


df_records, df_summary = evaluate_on_all(
    data_path=data_path,
    detector_class=PQRSTDetector,
    channel='i',
    window=0.25,
    max_records=None,
    skip_detector_points={
        'r_peaks': 1,  # пропускаем первый R-пик детектора
        'p_points': 2,  # пропускаем первые две P-точки детектора
        't_points': 1,  # пропускаем первый T-пик детектора
        'onsets': 4,  # пропускаем первую границу начала детектора
        'offsets': 4  # пропускаем первую границу конца детектора
    }
)

if not df_records.empty:
    df_records.to_csv('ludb_detection_results_channel_ii_detector_skipped.csv', index=False)
    print(f"\nСохранены результаты по {len(df_records)} записям")

df_summary.to_csv('ludb_detection_summary_channel_ii_detector_skipped.csv')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.2f}'.format)

print("\n" + df_summary.to_string())



# База данных QT
!wget -r -N -c -np https://physionet.org/files/qtdb/1.0.0/
data_path = "physionet.org/files/qtdb/1.0.0/"
record_name = "sel30"
# Полный путь к файлу без расширения
record_path = os.path.join(data_path, record_name)
record = wfdb.rdrecord(record_path)
sfreq = record.fs  # Частота дискретизации
channels = record.sig_name  # Названия каналов
data = record.p_signal.T  # Транспонируем для MNE (каналы x отсчеты)
# Создание информации для MNE
ch_types = ['ecg'] * len(channels)
info = mne.create_info(
    ch_names=channels,
    ch_types=ch_types,
    sfreq=sfreq
)
raw = mne.io.RawArray(data, info)
annotation = wfdb.rdann(record_path, 'pu')
# Получение аннотаций
annot_times = annotation.sample / sfreq
# Символы аннотаций (типы сокращений)
annot_labels = annotation.symbol
# Описание аннотаций (длительность - 0 для точечных аннотаций)
annot_durations = [0] * len(annot_times)
# Создание аннотаций MNE
annotations = mne.Annotations(
    onset=annot_times,
    duration=annot_durations,
    description=annot_labels
)


# Графики для QT
def plot_detector_vs_annotations(raw, detector_results, channel, record_path,
                                 tmin=0, tmax=10, fs=500):
    # Проверка наличия канала
    if channel not in raw.ch_names:
        print(f"Канал '{channel}' не найден. Доступные каналы: {raw.ch_names}")
        return None, None

    # Получение данных для указанного временного интервала
    data, times = raw.get_data(
        picks=channel,
        tmin=tmin,
        tmax=tmax,
        return_times=True
    )
    data = data[0]

    # Загружаем аннотации
    try:
        annotation = wfdb.rdann(record_path, channel)
    except:
        try:
            annotation = wfdb.rdann(record_path, 'pu0')
        except:
            try:
                annotation = wfdb.rdann(record_path, 'pu')
            except:
                print("Не удалось загрузить аннотации")
                return None, None

    sfreq = raw.info['sfreq']
    annot_times = annotation.sample / sfreq
    annot_labels = annotation.symbol

    mask = (annot_times >= tmin) & (annot_times <= tmax)
    annot_times_win = annot_times[mask]
    annot_labels_win = np.array(annot_labels)[mask]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True)

    ax1.plot(times, data, 'b-', linewidth=1, alpha=0.7, label=f'ЭКГ {channel.upper()}')

    detector_config = {
        'r_peaks': ('red', '^', 'Детектор R-пики', 'R', 120),
        'q_points': ('orange', 's', 'Детектор Q-точки', 'Q', 100),
        's_points': ('purple', 'D', 'Детектор S-точки', 'S', 100),
        'p_points': ('green', 'o', 'Детектор P-точки', 'P', 100),
        't_points': ('brown', 'v', 'Детектор T-точки', 'T', 100),
        'onsets': ('pink', '<', 'Детектор начала волн', '(', 80),
        'offsets': ('lightblue', '>', 'Детектор концы волн', ')', 80)
    }

    legend_added = set()

    for key, (color, marker, legend_name, text, size) in detector_config.items():
        if key in detector_results and len(detector_results[key]) > 0:
            points = detector_results[key]
            for idx in points:
                t = idx / fs
                if tmin <= t <= tmax:
                    seg_idx = np.argmin(np.abs(times - t))
                    y_val = data[seg_idx]

                    if legend_name not in legend_added:
                        ax1.scatter(t, y_val, color=color, marker=marker, s=size,
                                    edgecolors='black', linewidth=2, zorder=5,
                                    label=legend_name)
                        legend_added.add(legend_name)
                    else:
                        ax1.scatter(t, y_val, color=color, marker=marker, s=size,
                                    edgecolors='black', linewidth=2, zorder=5)

                    ax1.annotate(f'D:{text}', (t, y_val),
                                 xytext=(5, 5), textcoords='offset points',
                                 fontsize=8, color=color, fontweight='bold')

    ax1.set_ylabel('Амплитуда (мВ)', fontsize=12)
    ax1.set_title(f'РЕЗУЛЬТАТЫ ДЕТЕКТОРА', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8, ncol=2)

    ax2.plot(times, data, 'b-', linewidth=1, alpha=0.7, label=f'ЭКГ {channel.upper()}')

    annotation_config = {
        '(': {'color': 'lightgreen', 'marker': '<', 'label': 'P-onset', 'size': 80, 'text': '('},
        'p': {'color': 'green', 'marker': 'o', 'label': 'P-peak', 'size': 100, 'text': 'p'},
        ')': {'color': 'darkgreen', 'marker': '>', 'label': 'P-offset', 'size': 80, 'text': ')'},
        '[': {'color': 'pink', 'marker': '<', 'label': 'QRS-onset', 'size': 80, 'text': '['},
        'q': {'color': 'orange', 'marker': 's', 'label': 'Q-peak', 'size': 100, 'text': 'q'},
        'R': {'color': 'red', 'marker': '^', 'label': 'R-peak', 'size': 120, 'text': 'R'},
        'N': {'color': 'red', 'marker': '^', 'label': 'R-peak', 'size': 120, 'text': 'R'},
        's': {'color': 'purple', 'marker': 'D', 'label': 'S-peak', 'size': 100, 'text': 's'},
        ']': {'color': 'violet', 'marker': '>', 'label': 'QRS-offset', 'size': 80, 'text': ']'},
        't': {'color': 'brown', 'marker': 'v', 'label': 'T-peak', 'size': 100, 'text': 't'},
        '|': {'color': 'darkred', 'marker': '>', 'label': 'T-offset', 'size': 80, 'text': '|'}
    }

    legend_added_ann = set()

    for i, (t, label) in enumerate(zip(annot_times_win, annot_labels_win)):
        idx = np.argmin(np.abs(times - t))
        y_val = data[idx]

        if label in annotation_config:
            config = annotation_config[label]
            color = config['color']
            marker = config['marker']
            label_name = config['label']
            size = config['size']
            text = config['text']
        else:
            color = 'gray'
            marker = 'D'
            label_name = f'QT-DB: {label}'
            size = 60
            text = label

        if label_name not in legend_added_ann:
            ax2.plot(t, y_val, marker=marker, color=color, markersize=size // 10,
                     markeredgecolor='black', markeredgewidth=1.5,
                     linestyle='None', label=f'QT-DB: {label_name}')
            legend_added_ann.add(label_name)
        else:
            ax2.plot(t, y_val, marker=marker, color=color, markersize=size // 10,
                     markeredgecolor='black', markeredgewidth=1.5,
                     linestyle='None')

        ax2.annotate(text, (t, y_val),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8, color=color, fontweight='bold')

    ax2.set_xlabel('Время (с)', fontsize=12)
    ax2.set_ylabel('Амплитуда (мВ)', fontsize=12)
    ax2.set_title(f'ОРИГИНАЛЬНЫЕ АННОТАЦИИ QT-DB', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8, ncol=2)

    plt.tight_layout()
    plt.show()

    return fig, (ax1, ax2)


record_name = "sel853"
channel = 'ECG1'

record_path = os.path.join(data_path, record_name)
record = wfdb.rdrecord(record_path)
fs = record.fs

try:
    annotation = wfdb.rdann(record_path, channel)
except:
    try:
        annotation = wfdb.rdann(record_path, 'pu0')
    except:
        try:
            annotation = wfdb.rdann(record_path, 'pu')
        except:
            exit()

ch_types = ['ecg'] * len(record.sig_name)
info = mne.create_info(
    ch_names=record.sig_name,
    ch_types=ch_types,
    sfreq=fs
)

annot_times = annotation.sample / fs
annot_durations = [0] * len(annot_times)
mne_annotations = mne.Annotations(
    onset=annot_times,
    duration=annot_durations,
    description=annotation.symbol
)

raw = mne.io.RawArray(record.p_signal.T, info)
raw.set_annotations(mne_annotations)

data_for_detector, _ = raw.get_data(picks=channel, tmin=0, tmax=10, return_times=True)
ecg_segment = data_for_detector[0]

detector = PQRSTDetector(fs=fs)
results = detector.process(ecg_segment)
fig, axes = plot_detector_vs_annotations(
    raw=raw,
    detector_results=results,
    channel=channel,
    record_path=record_path,
    tmin=0,
    tmax=10,
    fs=fs
)


# Метрики для QT
def evaluate_on_qtdb(data_path, detector_class, channel='pu0', window=0.25, max_records=None):
    all_files = os.listdir(data_path)
    record_names = sorted(list(set([f.split('.')[0] for f in all_files if f.endswith('.dat')])))

    if max_records:
        record_names = record_names[:max_records]

    mapping = {
        'r_peaks': (['R', 'N'], 'R-пики'),
        'p_points': (['p'], 'P-точки'),
        't_points': (['t'], 'T-точки'),
        'onsets': (['(', '['], 'Начала волн (P+QRS+Т)'),
        'offsets': ([')', ']', '|'], 'Концы волн (P+QRS+T)')
    }

    all_results = []

    total_records_processed = 0
    records_processed = 0
    records_with_error = []

    total_stats = {
        'tp': {key: 0 for key in mapping.keys()},
        'fp': {key: 0 for key in mapping.keys()},
        'fn': {key: 0 for key in mapping.keys()},
        'errors': {key: [] for key in mapping.keys()}
    }
    total_stats['tp']['total'] = 0
    total_stats['fp']['total'] = 0
    total_stats['fn']['total'] = 0
    total_stats['errors']['total'] = []

    for record_name in tqdm(record_names, desc="Обработка записей QT-DB"):
        total_records_processed += 1
        try:
            record_path = os.path.join(data_path, record_name)

            record = wfdb.rdrecord(record_path)
            fs = record.fs

            if channel in record.sig_name:
                channel_idx = record.sig_name.index(channel)
                channel_name = channel
            elif channel == 'pu0' and '0' in record.sig_name:
                channel_idx = record.sig_name.index('0')
                channel_name = '0'
            elif channel == 'pu0' and 'I' in record.sig_name:
                channel_idx = record.sig_name.index('I')
                channel_name = 'I'
            else:
                channel_idx = 0
                channel_name = record.sig_name[0]

            ecg_data = record.p_signal[:, channel_idx]

            try:
                annotation = wfdb.rdann(record_path, channel)
            except:
                try:
                    annotation = wfdb.rdann(record_path, 'pu0')
                except:
                    try:
                        annotation = wfdb.rdann(record_path, 'pu')
                    except:
                        continue

            annot_times = annotation.sample / fs
            annot_labels = annotation.symbol

            qtdb_annotations = {}
            for time, label in zip(annot_times, annot_labels):
                if label not in qtdb_annotations:
                    qtdb_annotations[label] = []
                qtdb_annotations[label].append(time)

            for label in qtdb_annotations:
                qtdb_annotations[label] = np.sort(qtdb_annotations[label])

            detector = detector_class(fs=fs)
            detector_results = detector.process(ecg_data)

            det_points = {}
            for key in mapping.keys():
                if key in detector_results and len(detector_results[key]) > 0:
                    points = np.sort(detector_results[key] / fs)
                    det_points[key] = points
                else:
                    det_points[key] = np.array([])

            record_metrics = {'record': record_name}

            for det_key, (gt_symbols, name) in mapping.items():
                gt_points = []
                for sym in gt_symbols:
                    if sym in qtdb_annotations:
                        gt_points.extend(qtdb_annotations[sym])
                gt_points = np.sort(gt_points)

                det = det_points[det_key]

                if len(det) == 0 and len(gt_points) == 0:
                    tp = fp = fn = 0
                    errors = []
                elif len(det) == 0:
                    tp = 0
                    fp = 0
                    fn = len(gt_points)
                    errors = []
                elif len(gt_points) == 0:
                    tp = 0
                    fp = len(det)
                    fn = 0
                    errors = []
                else:
                    matched_det = np.zeros(len(det), dtype=bool)
                    matched_gt = np.zeros(len(gt_points), dtype=bool)
                    errors = []

                    for i, gt in enumerate(gt_points):
                        if not matched_gt[i]:
                            distances = np.abs(det - gt)
                            distances[matched_det] = np.inf

                            if len(distances) > 0:
                                min_idx = np.argmin(distances)
                                if distances[min_idx] <= window:
                                    matched_det[min_idx] = True
                                    matched_gt[i] = True
                                    errors.append((det[min_idx] - gt) * 1000)

                    tp = len(errors)
                    fp = len(det) - tp
                    fn = len(gt_points) - tp

                precision = tp / (tp + fp) * 100 if tp + fp > 0 else 0
                recall = tp / (tp + fn) * 100 if tp + fn > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

                mean_error = np.mean(errors) if errors else None

                record_metrics[f'{det_key}_tp'] = tp
                record_metrics[f'{det_key}_fp'] = fp
                record_metrics[f'{det_key}_fn'] = fn
                record_metrics[f'{det_key}_precision'] = precision
                record_metrics[f'{det_key}_recall'] = recall
                record_metrics[f'{det_key}_f1'] = f1
                record_metrics[f'{det_key}_mean_error'] = mean_error
                record_metrics[f'{det_key}_detected'] = len(det)
                record_metrics[f'{det_key}_ground_truth'] = len(gt_points)

                total_stats['tp'][det_key] += tp
                total_stats['fp'][det_key] += fp
                total_stats['fn'][det_key] += fn
                total_stats['errors'][det_key].extend(errors)
                total_stats['tp']['total'] += tp
                total_stats['fp']['total'] += fp
                total_stats['fn']['total'] += fn
                total_stats['errors']['total'].extend(errors)

            all_results.append(record_metrics)
            records_processed += 1

        except Exception as e:
            import traceback
            traceback.print_exc()
            records_with_error.append(record_name)
            continue

    if all_results:
        df_records = pd.DataFrame(all_results)
    else:
        df_records = pd.DataFrame()
        return df_records, pd.DataFrame()

    print(f"\nСтатистика по записям:")
    print(f"Обработано записей: {records_processed}")

    summary = {}

    for det_key, (_, name) in mapping.items():
        tp = total_stats['tp'][det_key]
        fp = total_stats['fp'][det_key]
        fn = total_stats['fn'][det_key]
        errors = total_stats['errors'][det_key]

        precision = tp / (tp + fp) * 100 if tp + fp > 0 else 0
        recall = tp / (tp + fn) * 100 if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        mean_error = np.mean(errors) if errors else None

        summary[det_key] = {
            'Тип': name,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'Precision %': round(precision, 2),
            'Recall %': round(recall, 2),
            'F1 %': round(f1, 2),
            'Ср.ошибка мс': round(mean_error, 2) if mean_error else None
        }

    total_tp = total_stats['tp']['total']
    total_fp = total_stats['fp']['total']
    total_fn = total_stats['fn']['total']
    total_errors = total_stats['errors']['total']

    total_precision = total_tp / (total_tp + total_fp) * 100 if total_tp + total_fp > 0 else 0
    total_recall = total_tp / (total_tp + total_fn) * 100 if total_tp + total_fn > 0 else 0
    total_f1 = 2 * total_precision * total_recall / (
                total_precision + total_recall) if total_precision + total_recall > 0 else 0

    summary['total'] = {
        'Тип': f'ВСЕГО (QT-DB, канал {channel})',
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn,
        'Precision %': round(total_precision, 2),
        'Recall %': round(total_recall, 2),
        'F1 %': round(total_f1, 2),
        'Ср.ошибка мс': round(np.mean(total_errors), 2) if total_errors else None
    }

    df_summary = pd.DataFrame(summary).T

    return df_records, df_summary


df_records, df_summary = evaluate_on_qtdb(
    data_path=data_path,
    detector_class=PQRSTDetector,
    channel='ECG1',
    window=0.25,
    max_records=None
)

if not df_records.empty:
    df_records.to_csv('qtdb_detection_results_pu0.csv', index=False)

df_summary.to_csv('qtdb_detection_summary_pu0.csv')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.2f}'.format)

print("\n" + df_summary.to_string())