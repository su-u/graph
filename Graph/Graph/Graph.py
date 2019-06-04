import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment as AS

data = AS.from_mp3("440Hz-64.mp3")
l = data.get_array_of_samples()[-1000:]

fs = data.frame_rate
print(fs)
d = 1.0 / fs
size = 1000

dt2 = np.fft.fft(l)
frq = np.fft.fftfreq(size, d)

plt.subplot(2,1,1)
plt.title("FFT")
plt.plot(l)
plt.subplot(2,1,2)
plt.title("FFT")
plt.plot(frq, abs(dt2))
plt.axis([0, 800, 0,max(abs(dt2)) + 100000])

plt.show()



#igfont = {'family':'Yu Gothic'}
#N = 1000
#dt = 0.01
#freq = 5
#amp = 1
#t = np.arange(0, N*dt, dt)
#f = amp * np.sin(2*np.pi*freq*t)
#f = t - np.floor(t)

#F = np.fft.fft(l)

##F_abs = np.abs(F)

##F_abs_amp = F_abs / N * 2
##F_abs_amp[0] = F_abs_amp[0] / 2

#fq = np.linspace(0, 1.0/dt, N)

#fig = plt.figure(figsize=(12, 4))

#ax2 = fig.add_subplot(121)
#plt.xlabel('time(sec)', fontsize=14)
#plt.ylabel('amplitude', fontsize=14)
#plt.title("時間領域のグラフ", **igfont)
##plt.plot(t, f)
#plt.plot(l)


#ax2 = fig.add_subplot(122)
#plt.subplot(1,2,2)
#plt.xlabel('freqency(Hz)', fontsize=14)
#plt.ylabel('amplitude', fontsize=14)
##plt.plot(fq[:int(N/2)+1], F_abs_amp[:int(N/2)+1])
#plt.plot(F)
#plt.title("周波数領域のグラフ", **igfont)
#plt.show()
