# From https://stackoverflow.com/questions/43963982/python-change-pitch-of-wav-file
# a script to modulate pitch
import wave
import numpy as np
import sys

up_rate = int(sys.argv[1])
print('up_rate', up_rate)

filename = 'nuro_vsg_loop_v01.wav'
wr = wave.open(filename, 'r')

print('getnchannels', wr.getnchannels())
print('getsampwidth', wr.getsampwidth())
print('getframerate', wr.getframerate())
print('getnframes', wr.getnframes())
print('getcomptype', wr.getcomptype())
print('getcompname', wr.getcompname())
print('getparams', wr.getparams())

# Set the parameters for the output file.
par = list(wr.getparams())
print('par', par)
par[3] = 0  # The number of samples will be set by writeframes.
par = tuple(par)
ww = wave.open('pitch_%d.wav' % (up_rate), 'w')
ww.setparams(par)

WAV_FREQ = 145  # frequency found by another program
up_freq = int(WAV_FREQ * 0.01 * up_rate)
print('up_freq', up_freq)

# fr = 5
# sz = wr.getframerate()//fr  # Read and process 1/fr second at a time.
# # A larger number for fr means less reverb.
# c = int(wr.getnframes()/sz)  # count of the whole file
# shift = up_freq//fr  #

# hack
sz = wr.getnframes()
c = 1
shift = int(up_freq * (wr.getnframes() / wr.getframerate()))
for num in range(c):

    da = np.fromstring(wr.readframes(sz), dtype=np.int16)
    # left, right = da[0::2], da[1::2]  # left and right channel
    mono = da

    # lf, rf = np.fft.rfft(left), np.fft.rfft(right)
    monof = np.fft.rfft(mono)

    # lf, rf = np.roll(lf, shift), np.roll(rf, shift)
    monof = np.roll(monof, shift)

    # lf[0:shift], rf[0:shift] = 0, 0
    # monof[0:shift] = 0

    # nl, nr = np.fft.irfft(lf), np.fft.irfft(rf)
    nmono = np.fft.irfft(monof)

    # ns = np.column_stack((nl, nr)).ravel().astype(np.int16)
    ns = np.column_stack(nmono).ravel().astype(np.int16)

    ww.writeframes(ns.tostring())

wr.close()
ww.close()