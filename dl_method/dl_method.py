import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from typing import List
import mne
import os
import wfdb
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.signal import resample

class Preprocessor:
    def __init__(self, fs=250, wavelet='db6', level=4, lowcut=0.5, highcut=40.0):
        self.fs = fs
        self.wavelet = wavelet
        self.level = level
        self.lowcut = lowcut
        self.highcut = highcut

    def wavelet_denoise(self, signal):
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        denoised_coeffs = [coeffs[0]]
        for detail in coeffs[1:]:
            sigma = np.median(np.abs(detail)) / 0.6745
            denoised_coeffs.append(pywt.threshold(detail, threshold, mode='soft'))
        
        denoised = pywt.waverec(denoised_coeffs, self.wavelet)
    
            
        if len(denoised) > len(signal):
            denoised = denoised[:len(signal)]
        elif len(denoised) < len(signal):
            denoised = np.pad(denoised, (0, len(signal) - len(denoised)), mode='edge')
        return denoised

    def bandpass_filter(self, signal, lowcut=0.5, highcut=40.0, order=3):
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        
        b, a = butter(order, [low, high], btype='bandpass')
        return filtfilt(b, a, signal)

    def process(self, signal):
        processed = []
        
        for lead in signal:
            lead_changed = self.wavelet_denoise(lead)
            lead_changed = self.bandpass_filter(lead_changed)
            processed.append(lead_changed)
        return np.stack(processed)


class LabelProcessor:
    def __init__(self, window=37):
        self.window = window
    def extend(self, label):
        extended = np.zeros_like(label)
        for i, val in enumerate(label):
            if val == 0:
                continue
            if isinstance(self.window, dict):
                w = self.window.get(int(val), 5)
                if isinstance(w, (list, tuple)):
                    l, r = w[0], w[1]
                else:
                    l, r = w, w
            else:
                l, r = self.window, self.window
            start = max(0, i - l)
            end = min(len(label), i + r + 1)
            extended[start:end] = val
        return extended



class UNetECGCNN(nn.Module):
    def __init__(self, num_classes: int = 10, num_channels=12):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(num_channels, 16, kernel_size=31)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.enc2 = self._conv_block(16, 32, kernel_size=25)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.enc3 = self._conv_block(32, 64, kernel_size=19)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Bottom
        self.bottom = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=13, padding=6),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=13, padding=6),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec3 = self._conv_block(128 + 64, 64, kernel_size=19)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec2 = self._conv_block(64 + 32, 32, kernel_size=25)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec1 = self._conv_block(32 + 16, 16, kernel_size=31)
        
        # Final
        self.final_conv = nn.Conv1d(16, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int, kernel_size: int):
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_length = x.shape[-1]
        
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # Bottom
        b = self.bottom(p3)
        
        # Decoder
        u3 = self.up3(b)
        if u3.shape[-1] != e3.shape[-1]:
            u3 = F.interpolate(u3, size=e3.shape[-1], mode='linear', align_corners=False)
        cat3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(cat3)
        
        u2 = self.up2(d3)
        if u2.shape[-1] != e2.shape[-1]:
            u2 = F.interpolate(u2, size=e2.shape[-1], mode='linear', align_corners=False)
        cat2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(cat2)
        
        u1 = self.up1(d2)
        if u1.shape[-1] != e1.shape[-1]:
            u1 = F.interpolate(u1, size=e1.shape[-1], mode='linear', align_corners=False)
        cat1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(cat1)
        
        # Final
        logits = self.final_conv(d1)
        
        return self.softmax(logits)


class UNetECGLSTM(nn.Module):
    def __init__(self, num_classes: int = 10, num_channels=12):
        super().__init__()
        # Encoder
        self.enc1 = UNetECGCNN._conv_block(num_channels, 16, 31)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = UNetECGCNN._conv_block(16, 32, 25)
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = UNetECGCNN._conv_block(32, 64, 19)
        self.pool3 = nn.MaxPool1d(2)

        # Bottom
        self.lstm1 = nn.LSTM(64, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 128, batch_first=True)

        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec3 = UNetECGCNN._conv_block(128 + 64, 64, 19)
        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec2 = UNetECGCNN._conv_block(64 + 32, 32, 25)
        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec1 = UNetECGCNN._conv_block(32 + 16, 16, 31)

        self.final = nn.Conv1d(16, num_classes, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_length = x.shape[-1]
        
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottom
        seq = p3.permute(0, 2, 1)
        l1, _ = self.lstm1(seq)
        l2, _ = self.lstm2(l1)
        b = l2.permute(0, 2, 1)

        # Decoder
        u3 = self.up3(b)
        if u3.shape[-1] != e3.shape[-1]:
            u3 = F.interpolate(u3, size=e3.shape[-1], mode='linear', align_corners=False)
        d3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(d3)
        
        u2 = self.up2(d3)
        if u2.shape[-1] != e2.shape[-1]:
            u2 = F.interpolate(u2, size=e2.shape[-1], mode='linear', align_corners=False)
        d2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(d2)
        
        u1 = self.up1(d2)
        if u1.shape[-1] != e1.shape[-1]:
            u1 = F.interpolate(u1, size=e1.shape[-1], mode='linear', align_corners=False)
        d1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return self.softmax(out)


class MultiChannelLoss(nn.Module):
    def __init__(self, num_channels=2, ignore_index=0):
        super().__init__()
        self.num_channels = num_channels
        self.nll = nn.NLLLoss(ignore_index=ignore_index)

    def forward(self, pred, target):
        total_loss = 0.0
        for ch in range(self.num_channels):
            log_probs = torch.log(pred[:, ch] + 1e-8)
            loss_ch = self.nll(log_probs, target[:, ch])
            total_loss += loss_ch
        return total_loss / self.num_channels


class FixedThreshold:
    def __init__(self, thr=0.5):
        self.thr = thr

    @staticmethod
    def find_fragments(binary):
        indices = np.where(binary == 1)[0]
        if len(indices) == 0:
            return []
        fragments = []
        start = indices[0]
        prev = indices[0]
        for idx in indices[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                fragments.append((start, prev))
                start = idx
                prev = idx
        fragments.append((start, prev))
        return fragments

    def detect(self, prob_vector):
        binary = (prob_vector > self.thr).astype(int)
        fragments = self.find_fragments(binary)
        midpoints = [(s + e) // 2 for s, e in fragments]
        return midpoints


class DTAA:
    def __init__(self, Fs=250, thr1=0.5, search_margin_ratio=0.2):
        self.Fs = Fs
        self.thr1 = thr1
        self.margin_sample = int(search_margin_ratio * self.Fs)
    

    def detect(self, P):
        ft = FixedThreshold(self.thr1)
        QRS_p = ft.detect(P)

        QRS_xin = []

        if len(QRS_p) == 0:
            return []

        M = len(P)

        # Step A
        if QRS_p[0] > self.Fs:
            r_border = min(M, QRS_p[0] + self.margin_sample)
            window = P[:r_border]
            thr2 = self.thr1 - 0.1
            while thr2 > 0:
                ft2 = FixedThreshold(thr2)
                new_pts = ft2.detect(window)
                if len(new_pts) > 0:
                    QRS_xin.extend(new_pts)
                    break
                else:
                    thr2 -= 0.1
        else:
            QRS_xin.append(QRS_p[0])

        # Step B
        for i in range(1, len(QRS_p) - 1):
            if QRS_p[i] - QRS_xin[-1] > 1.2 * self.Fs:
                thr3 = self.thr1 - 0.1
                a = QRS_xin[-1] + self.margin_sample
                b = QRS_p[i] - self.margin_sample
                if b <= a:
                    continue
                window = P[a:b]
                while thr3 > 0:
                    ft3 = FixedThreshold(thr3)
                    new_pts = ft3.detect(window)
                    if len(new_pts) > 0:
                        QRS_xin.extend([a + pt for pt in new_pts])
                        break
                    else:
                        thr3 -= 0.1
            else:
                QRS_xin.append(QRS_p[i])

        # Step C
        if QRS_p[-1] < M - self.Fs:
            thr4 = self.thr1 - 0.1
            a = QRS_p[-1] + self.margin_sample
            window = P[a:]
            while thr4 > 0:
                ft4 = FixedThreshold(thr4)
                new_pts = ft4.detect(window)
                if len(new_pts) > 0:
                    QRS_xin.extend([a + pt for pt in new_pts])
                    break
                else:
                    thr4 -= 0.1

        cleaned = []
        for pt in sorted(QRS_xin):
            if len(cleaned) == 0:
                cleaned.append(pt)
            else:
                if pt - cleaned[-1] < 0.3 * self.Fs:
                    if P[pt] > P[cleaned[-1]]:
                        cleaned[-1] = pt
                else:
                    cleaned.append(pt)

        return cleaned


class ChannelDetector:
    def __init__(self, model, target_channel='pu0', fs=250, orig_fs=250, device='cuda', thr1=0.5, window_size=10, stride=9):
        self.model = model
        self.device = device
        self.channel_names = [target_channel]
        self.target_idx = self.channel_names.index(target_channel)
        self.dtaa = DTAA(fs, thr1=thr1)
        self.fs = fs
        self.preprocessor = Preprocessor(fs=fs)
        self.window_size = window_size * fs
        self.stride = stride * fs
        self.orig_fs = orig_fs
    
    @torch.no_grad()
    def predict_full_probs(self, ecg):
        self.model.eval()
        if self.orig_fs != 250:
            if self.orig_fs % 250 != 0:
                raise TypeError(f"The sampling rate is not a multiple of 250")
            target_fs = 250

            L_orig = ecg.shape[1]
            L_new = int(L_orig * target_fs / self.orig_fs)

            ecg = resample(ecg, L_new, axis=1)


        signal = self.preprocessor.process(ecg)

        L = signal.shape[1]
        C = self.model.num_classes

        final_probs = np.zeros((C, L))
        counts = np.zeros(L)

        weights = np.hanning(self.window_size)

        for start in range(0, L, self.stride):
            end = start + self.window_size

            if end > L:
                pad = end - L
                segment = np.pad(signal[:, start:L], ((0, 0), (0, pad)))
            else:
                segment = signal[:, start:end]

            x = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to(self.device)

            pred = self.model(x)

            #EK
            ek_logits = pred.mean(dim=1)[0]
            probs = ek_logits.cpu().numpy()
            valid_len = min(self.window_size, L - start)

            final_probs[:, start:start+valid_len] += probs[:, :valid_len] * weights[:valid_len]
            counts[start:start+valid_len] += weights[:valid_len]

        final_probs /= (counts + 1e-8)
        return final_probs
  
  

    def refine_wave(self, start, peak, end, is_t_wave=False):
        refined_start = []
        refined_peak = []
        refined_end = []

        for p in peak:
            s_candidates = [s for s in start if s < p]
            if not s_candidates:
                continue
            s = max(s_candidates)

            e_candidates = [e for e in end if e > p]
            if not e_candidates:
                continue
            e = min(e_candidates)

            duration = e - s

            if is_t_wave:
                min_dur = 0.04 * self.fs
                max_dur = 0.60 * self.fs
            else:
                min_dur = 0.02 * self.fs
                max_dur = 0.50 * self.fs

            if not (min_dur < duration < max_dur):
                continue

            refined_start.append(s)
            refined_peak.append(p)
            refined_end.append(e)

        return refined_start, refined_peak, refined_end




    @torch.no_grad()
    def detect(self, ecg):
        probs = self.predict_full_probs(ecg)


        detections = []
        for class_probs in probs:
            a = self.dtaa.detect(class_probs)
            detections.append(a)
        


        qrs_end = detections[6]
        t_start = detections[7]
        t_peak = detections[8]

        # P
        p_start, p_peak, p_end = self.refine_wave(
            detections[1], detections[2], detections[3]
        )

        # QRS
        qrs_start, qrs_peak, qrs_end = self.refine_wave(
            detections[4], detections[5], detections[6]
        )

        # T
        t_start, t_peak, t_end = self.refine_wave(
            detections[7], detections[8], detections[9], is_t_wave=True
        )

        detections[1] = p_start
        detections[2] = p_peak
        detections[3] = p_end

        detections[4] = qrs_start
        detections[5] = qrs_peak
        detections[6] = qrs_end

        detections[7] = t_start
        detections[8] = t_peak
        detections[9] = t_end


        return detections



def collate_fn(batch):
    if len(batch[0]) == 2:
        signals, labels = zip(*batch)
        max_len = max(s.shape[1] for s in signals)
        if max_len % 8 != 0:
            max_len = ((max_len // 8) + 1) * 8
        padded_signals, padded_labels = [], []
        for s, l in zip(signals, labels):
            if s.shape[1] < max_len:
                s = F.pad(s, (0, max_len - s.shape[1]))
            if l.shape[1] < max_len:
                l = F.pad(l, (0, max_len - l.shape[1]), value=0)
            padded_signals.append(s)
            padded_labels.append(l)
        return torch.stack(padded_signals), torch.stack(padded_labels)

    elif len(batch[0]) == 4:
        x_orig, y_orig, x_aug, y_aug = zip(*batch)
        max_len = max(max(s.shape[1] for s in x_orig),
                      max(s.shape[1] for s in x_aug))
        if max_len % 8 != 0:
            max_len = ((max_len // 8) + 1) * 8
        def pad_list(lst):
            padded = []
            for s in lst:
                if s.shape[1] < max_len:
                    s = F.pad(s, (0, max_len - s.shape[1]))
                padded.append(s)
            return padded
        x_orig = torch.stack(pad_list(x_orig))
        y_orig = torch.stack(pad_list(y_orig))
        x_aug = torch.stack(pad_list(x_aug))
        y_aug = torch.stack(pad_list(y_aug))
        return x_orig, y_orig, x_aug, y_aug


class Evaluator:
    def __init__(self, fs=250, tolerance_ms=150):
        self.fs = fs
        self.tol = int(fs * tolerance_ms / 1000)

    def match(self, predicted, truth):
        predicted = np.array(predicted)
        truth = np.array(truth)

        TP = 0
        errors = []
        used = set()

        for p in predicted:
            distances = np.abs(truth - p)
            idx = np.argmin(distances)

            if distances[idx] <= self.tol and idx not in used:
                TP += 1
                used.add(idx)
                errors.append(p - truth[idx])

        FP = len(predicted) - TP
        FN = len(truth) - TP

        return TP, FP, FN, errors

    def compute_metrics(self, predicted, truth):
        tp, fp, fn, errors = self.match(predicted, truth)

        se = tp / (tp + fn) if (tp + fn) > 0 else 0
        pp = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (se * pp) / (se + pp) if (se + pp) > 0 else 0

        E = np.mean(errors) if len(errors) > 0 else None
        ST = np.std(errors) if len(errors) > 0 else None

        return {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Sensitivity": se,
            "PositivePredictivity": pp,
            "F1": f1,
            "MeanError": E,
            "StdError": ST
        }

class MultiChannelQTDataset(Dataset):
    def __init__(self, path, preprocessor, records, window=37):
        self.records = records
        self.path = path
        self.pre = preprocessor
        self.channels = ['pu0', 'pu1']
        self.label_processor = LabelProcessor(window=window)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        record_path = os.path.join(self.path, record)

        sig, _ = wfdb.rdsamp(record_path)
        sig = sig.T
        sig = self.pre.process(sig)

        L = sig.shape[1]

        multi_channel_labels = np.zeros((len(self.channels), L), dtype=np.int64)

        for ch_idx, ch_name in enumerate(self.channels):
            try:
                ann = wfdb.rdann(record_path, ch_name)


                start = None
                peak = None
                wave = None

                for s, sym in zip(ann.sample, ann.symbol):
                    if sym == '(':
                        start = int(s)
                        peak = None
                        wave = None
                    elif sym in ['p', 'N', 't']:
                        peak = int(s)
                        wave = sym
                    elif sym == ')':
                        end = int(s)

                        if start is None or peak is None or wave is None:
                            start = peak = wave = None
                            continue

                        if not (0 <= start < L and 0 <= peak < L and 0 <= end < L):
                            start = peak = wave = None
                            continue

                        if wave == 'p':
                            multi_channel_labels[ch_idx, start] = 1
                            multi_channel_labels[ch_idx, peak] = 2
                            multi_channel_labels[ch_idx, end] = 3
                        elif wave == 'N':
                            multi_channel_labels[ch_idx, start] = 4
                            multi_channel_labels[ch_idx, peak] = 5
                            multi_channel_labels[ch_idx, end] = 6
                        elif wave == 't':
                            multi_channel_labels[ch_idx, start] = 7
                            multi_channel_labels[ch_idx, peak] = 8
                            multi_channel_labels[ch_idx, end] = 9

                        start = None
                        peak = None
                        wave = None
            except:
                continue

        for ch_idx in range(len(self.channels)):
            multi_channel_labels[ch_idx] = self.label_processor.extend(multi_channel_labels[ch_idx])

        return (
            torch.tensor(sig, dtype=torch.float32), 
            torch.tensor(multi_channel_labels, dtype=torch.long)
        )
class WindowedECGDataset(Dataset):
    def __init__(self, path, records, preprocessor,
                 window_sec=10, stride_sec=9, fs=500, window=37):
        
        self.path = path
        self.records = records
        self.pre = preprocessor
        self.fs = fs
        
        self.window_size = window_sec * fs
        self.stride = stride_sec * fs
        
        self.channels = ['pu0', 'pu1']
        self.label_processor = LabelProcessor(window=window)

        self.samples = [] # (record, start, end)

        self._prepare_segments()

    def _prepare_segments(self):
        for record in self.records:
            record_path = os.path.join(self.path, record)

            sig, _ = wfdb.rdsamp(record_path)
            L = sig.shape[0]

            for start in range(0, L, self.stride):
                end = start + self.window_size
                self.samples.append((record, start, end))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        record, start, end = self.samples[idx]
        record_path = os.path.join(self.path, record)

        sig, _ = wfdb.rdsamp(record_path)
        sig = sig.T
        sig = self.pre.process(sig)

        L = sig.shape[1]

        if end > L:
            pad = end - L
            segment = np.pad(sig[:, start:L], ((0,0),(0,pad)))
        else:
            segment = sig[:, start:end]


        labels_full = np.zeros((len(self.channels), L), dtype=np.int64)

        for ch_idx, ch_name in enumerate(self.channels):
            try:
                ann = wfdb.rdann(record_path, ch_name)

                start_w, peak, wave = None, None, None

                for s, sym in zip(ann.sample, ann.symbol):
                    if sym == '(':
                        start_w = int(s)
                    elif sym in ['p', 'N', 't']:
                        peak = int(s)
                        wave = sym
                    elif sym == ')':
                        end_w = int(s)

                        if None in (start_w, peak, wave):
                            continue

                        if wave == 'p':
                            labels_full[ch_idx, start_w] = 1
                            labels_full[ch_idx, peak] = 2
                            labels_full[ch_idx, end_w] = 3
                        elif wave == 'N':
                            labels_full[ch_idx, start_w] = 4
                            labels_full[ch_idx, peak] = 5
                            labels_full[ch_idx, end_w] = 6
                        elif wave == 't':
                            labels_full[ch_idx, start_w] = 7
                            labels_full[ch_idx, peak] = 8
                            labels_full[ch_idx, end_w] = 9

                        start_w, peak, wave = None, None, None
            except Exception as e:
                print(f"{record} {ch_name}: {e}")

        for ch in range(len(self.channels)):
            labels_full[ch] = self.label_processor.extend(labels_full[ch])
        if end > L:
            pad = end - L
            segment_labels = np.pad(labels_full[:, start:L], ((0,0),(0,pad)))
        else:
            segment_labels = labels_full[:, start:end]

        return (
            torch.tensor(segment, dtype=torch.float32),
            torch.tensor(segment_labels, dtype=torch.long)
        )
    
class MultiChannelLinearEnsemble(nn.Module):
    def __init__(self, num_classes: int = 10, num_channels: int = 2):
        super().__init__()
        
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.cnn_model = UNetECGCNN(num_classes=num_classes, num_channels=1)
        self.lstm_model = UNetECGLSTM(num_classes=num_classes, num_channels=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, seq_len = x.shape
        
        outputs = []

        for ch in range(num_channels):
            x_ch = x[:, ch:ch+1, :]
            cnn_out = self.cnn_model(x_ch)
            lstm_out = self.lstm_model(x_ch)
            ensemble_out = 0.5 * cnn_out + 0.5 * lstm_out

            outputs.append(ensemble_out)

        stacked = torch.stack(outputs, dim=1)

        return stacked

def train_model(data_path, train_records, val_records=None, 
                num_epochs=60, batch_size=64, learning_rate=0.005, 
                window={1: (10, 5), 2: (4, 4), 3: (5, 10), 4: (4, 2), 5: (2, 2), 6: (2, 4), 7: (10, 5), 8: (6, 6), 9: (5, 10)}, 
                wavelet='db6', level=4, 
                lowcut=0.5, highcut=40.0):

    preprocessor = Preprocessor(fs=250, 
                                wavelet=wavelet, level=level, 
                                lowcut=lowcut, highcut=highcut)

    train_dataset = WindowedECGDataset(data_path, train_records, preprocessor, 
                                          window_sec=10, stride_sec=9, fs=250, window=window)
    val_dataset   = WindowedECGDataset(data_path, val_records,   preprocessor, 
                                          window_sec=10, stride_sec=9, fs=250, window=window)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, 
                              collate_fn=collate_fn, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MultiChannelLinearEnsemble(num_classes=10, num_channels=2).to(device)


    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=learning_rate, 
                                momentum=0.9)
    criterion = MultiChannelLoss(num_channels=2)

    best_val_loss = float('inf')
    patience = 20
    counter = 0
    best_model_path = 'model.pth'

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for x, y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)

        print(f'Epoch {epoch+1:2d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"The best model has been saved, val_loss = {best_val_loss:.6f}")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break

    model.load_state_dict(torch.load(best_model_path))
    print(f"The training is completed. The best val_loss = {best_val_loss:.6f}")

    return model

def extract_true_peaks_all(record_path, channel='pu0', num_classes=10, orig_fs=250):
    true_positions = {i: [] for i in range(1, num_classes)}
    
    try:
        ann = wfdb.rdann(record_path, channel)
        
        start = None
        peak = None
        wave = None
        if orig_fs != 250:
            samples = [int(s * 250 / orig_fs) for s in ann.sample]
        else:
            samples = ann.sample
        for s, sym in zip(samples, ann.symbol):
            if sym == '(':
                start = int(s)
                peak = None
                wave = None
            
            elif sym in ['N', 't', 'p']:
                peak = int(s)
                wave = sym
            

            elif sym == ')':
                end = int(s)
                
                if start is None or peak is None or wave is None:
                    start = peak = wave = None
                    continue
                if wave == 'p':
                    true_positions[1].append(start)
                    true_positions[2].append(peak) 
                    true_positions[3].append(end) 
                
                elif wave == 'N':
                    true_positions[4].append(start)
                    true_positions[5].append(peak) 
                    true_positions[6].append(end) 
                
                elif wave == 't':
                    true_positions[7].append(start) 
                    true_positions[8].append(peak) 
                    true_positions[9].append(end) 

                start = None
                peak = None
                wave = None

        for class_id in true_positions:
            true_positions[class_id] = np.array(sorted(true_positions[class_id]))
        

    except Exception as e:
        print(f"Error uploading annotations for the channel {channel}: {e}")
    
    return true_positions


def evaluate_model_all_classes(model, data_path, val_records, target_channel='pu0', thr1=0.5, orig_fs=250, text=False):
    device = next(model.parameters()).device
    preprocessor = Preprocessor(fs=250)
    evaluator = Evaluator(fs=250, tolerance_ms=150)
    detector = ChannelDetector(model, target_channel=target_channel, fs=250, thr1=thr1, orig_fs=orig_fs, device=device)
    
    all_metrics = {class_id: [] for class_id in range(1, 10)}
    
    for record_name in tqdm(val_records, desc="Оценка"):
        record_path = os.path.join(data_path, record_name)
        
        signal, _ = wfdb.rdsamp(record_path)
        signal = signal.T
        processed = preprocessor.process(signal)
        
        predictions = detector.detect(processed)
        
        true_positions = extract_true_peaks_all(record_path, channel=target_channel, orig_fs=orig_fs)
        
        for class_id in range(1, 10):
            pred = predictions[class_id]
            truth = true_positions.get(class_id, [])
            
            if len(pred) > 0 and len(truth) > 0:
                metrics = evaluator.compute_metrics(pred, truth)
                all_metrics[class_id].append(metrics)

    
    class_names = {
        1: "P_start", 2: "P_peak", 3: "P_end",
        4: "QRS_start", 5: "QRS_peak", 6: "QRS_end",
        7: "T_start", 8: "T_peak", 9: "T_end"
    }
    metrics_name = ["TP", "FP", "FN", "Sensitivity", "PositivePredictivity", "F1", "MeanError", "StdError"]
    full_metrics_mean = {}
    for class_id in range(1, 10):
        if all_metrics[class_id]:
            if text:
                print(f"\n{class_names[class_id]} (класс {class_id}):")
            metrics_mean = {}
            for metric in metrics_name:
                values = [m[metric] for m in all_metrics[class_id] if m[metric] is not None and not np.isnan(m[metric])]
                if text:
                    print(f"  {metric}: {np.mean(values):.3f} ± {np.std(values):.3f}")
                metrics_mean[metric] = np.mean(values)
            full_metrics_mean[class_names[class_id]] = metrics_mean
    
    return all_metrics, full_metrics_mean

def plot_ecg(record_path, signal_channel=None, annot_channel='pu0',
                                 tmin=0, tmax=10, res=None):
    
    record = wfdb.rdrecord(record_path)
    sfreq = record.fs
    data = record.p_signal.T
    channels = record.sig_name

    ch_types = ['ecg'] * len(channels)
    info = mne.create_info(ch_names=channels, ch_types=ch_types, sfreq=sfreq)
    raw = mne.io.RawArray(data, info)
    if signal_channel is None:
        signal_channel = record.sig_name[0]


    data, times = raw.get_data(
        picks=[signal_channel],
        tmin=tmin,
        tmax=tmax,
        return_times=True
    )

    annotation = wfdb.rdann(record_path, annot_channel)
    annot_times = annotation.sample / annotation.fs
    annot_labels = annotation.symbol

    waves = []
    start = None
    peak = None
    wave_type = None

    for s_sec, sym in zip(annot_times, annot_labels):
        if sym == '(':
            start = s_sec
            peak = None
            wave_type = None
        elif sym in ['p', 'N', 't']:
            peak = s_sec
            wave_type = sym
        elif sym == ')':
            end = s_sec
            if start is not None and peak is not None and wave_type is not None:
                waves.append((start, peak, end, wave_type))
            start = peak = wave_type = None

    fig, ax = plt.subplots(figsize=(30, 14))

    ax.plot(times, data[0], 'b-', linewidth=1, label=f'ECG {signal_channel}')

    style_map = {
        'p': {'color': 'purple', 'label': 'P'},
        'N': {'color': 'orange', 'label': 'QRS'},
        't': {'color': 'brown',  'label': 'T'},
    }

    plotted = set()

    for start_t, peak_t, end_t, wtype in waves:

        if end_t < tmin or start_t > tmax:
            continue

        style = style_map.get(wtype, {'color': 'purple', 'label': wtype.upper()})

        for t_val, label_suffix in [(start_t, 'start'), (peak_t, 'peak'), (end_t, 'end')]:
            if not (tmin <= t_val <= tmax):
                continue

            idx = np.argmin(np.abs(times - t_val))
            y_val = data[0, idx]

            lbl = f"{style['label']} {label_suffix}"

            if lbl not in plotted:
                ax.plot(t_val, y_val, 'o', color=style['color'],
                        markersize=10, label=lbl)
                plotted.add(lbl)
            else:
                ax.plot(t_val, y_val, 'o', color=style['color'], markersize=10)

            ax.annotate(label_suffix[0].upper(), (t_val, y_val),
                        xytext=(0, 8), textcoords='offset points',
                        fontsize=8, color=style['color'], ha='center')


    if res is not None and len(res) == 9:
        pred_styles = [
            ('P start',    'lightblue'),
            ('P peak',     'blue'),
            ('P end',      'darkblue'),
            ('QRS start',  'pink'),
            ('QRS peak',   'red'),
            ('QRS end',    'darkred'),
            ('T start',    'lightgreen'),
            ('T peak',     'green'),
            ('T end',      'darkgreen')
        ]

        plotted_pred = set()

        for (label, color), pred_idxs in zip(pred_styles, res):
            for idx in pred_idxs:
                t_val = idx / 250.0
                if not (tmin <= t_val <= tmax):
                    continue

                time_idx = np.argmin(np.abs(times - t_val))
                y_val = data[0, time_idx]

                if label not in plotted_pred:
                    ax.plot(t_val, y_val, marker='>', color=color,
                            markersize=12, label=f'Pred {label}')
                    plotted_pred.add(label)
                else:
                    ax.plot(t_val, y_val, marker='>', color=color,
                            markersize=12)

    ax.set_xlabel('Time (s)', fontsize=20)
    ax.set_ylabel('Amplitude', fontsize=20)
    ax.set_title(f'ECG - {signal_channel.upper()} | True vs Predicted (250 Hz)', fontsize=20)
    ax.grid(True, alpha=0.3)

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), fontsize=20, frameon=True, ncol=5)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

    return fig, ax

def make_detection(record_path, model, channel='pu0'):
    device = next(model.parameters()).device
    record = wfdb.rdrecord(record_path)
    sfreq = record.fs 

    preprocessor = Preprocessor(fs=250, wavelet='db6', level=4, lowcut=0.5, highcut=40)
    detector = ChannelDetector(model, target_channel=channel, fs=250, orig_fs=sfreq, device=device)
    signal, _ = wfdb.rdsamp(record_path)
    signal = signal.T
    processed = preprocessor.process(signal)
    res = detector.detect(processed)
    return res[1:]
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiChannelLinearEnsemble(num_classes=10, num_channels=2).to(device)
model.load_state_dict(torch.load('best_final.pth', map_location='cpu'))

# record_path = '...'
# results = make_detection(record_path, model)
# plot_ecg(record_path, res=results)