import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment as AS

igfont = {'family':'Yu Gothic'}

#data = AS.from_mp3("440Hz-64.mp3")
data = AS.from_wav("a_1.wav")
l = data.get_array_of_samples()

#データ読み込み時
fs = data.frame_rate
print(fs)
d = 1.0 / fs
size = 13078

dt2 = np.fft.fft(l)
frq = np.fft.fftfreq(size, d)


plt.subplot(2,1,1)
plt.title("時間領域", **igfont)
plt.plot(l)
plt.subplot(2,1,2)
plt.title("周波数領域", **igfont)
plt.plot(frq, abs(dt2))
plt.axis([0, fs/10.0, 0,max(abs(dt2)) + 100000])

plt.show()


## 簡単な信号の作成
#N = 180 # サンプル数
#dt = 0.01 # サンプリング周期(sec):100ms =>サンプリング周波数100Hz
#freq = 5 # 周波数(10Hz) =>正弦波の周期0.1sec
#amp = 1 # 振幅
#t = np.arange(0, N*dt, dt) # 時間軸
#f = amp * np.sin(2*np.pi*freq*t) # 信号（周波数10、振幅1の正弦波）

#N = 100 # データ数
#n = np.arange(N)
#f1 = 5 # 周期①
#f2 = 10 # 周期②

#a1 = 1 # 振幅①
#a2 = 5 # 振幅②
#f = a1 * np.sin(f1 * 2 * np.pi * (n/N)) + a2 * np.sin(f2 * 2 * np.pi * (n/N)) 

### 高速フーリエ変換(FFT)
#F = np.fft.fft(f) #

## FFTの複素数結果を絶対に変換
#F_abs = np.abs(F)
## 振幅をもとの信号に揃える
#F_abs_amp = F_abs / N * 2 # 交流成分はデータ数で割って2倍
#F_abs_amp[0] = F_abs_amp[0] / 2 # 直流成分（今回は扱わないけど）は2倍不要

## 周波数軸のデータ作成
#fq = np.linspace(0, 1.0/dt, N) # 周波数軸　linspace(開始,終了,分割数)

## グラフ表示
#fig = plt.figure(figsize=(12, 7))
## 信号のグラフ（時間軸）
##plt.title("時間領域", **igfont)
#ax2 = fig.add_subplot(211)
#plt.xlabel('time(sec)', fontsize=14)
#plt.ylabel('amplitude', fontsize=14)
#plt.plot(f)

## FFTのグラフ（周波数軸）
#ax2 = fig.add_subplot(212)
##plt.title("周波数領域", **igfont)
#plt.xlabel('freqency(Hz)', fontsize=14)
#plt.ylabel('amplitude', fontsize=14)
#plt.plot(fq[:int(N/2)+1], F_abs_amp[:int(N/2)+1]) # ナイキスト定数まで表示

#plt.show()