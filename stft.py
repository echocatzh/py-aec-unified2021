# shmzhang@aslp, 2021-04
import numpy as np


def dataLength(tau, stftshift, fftsize):
    len = (tau-1)*stftshift+fftsize
    return len


def numFrames(len, stftshift):
    tau = len//stftshift
    return tau


def stft(x, stftshift, fftsize):
    """Short time DFT

    Args:
        x (np.ndarray): input signal
        stftshift (int): time shift
        fftsize (int): fft window size

    Returns:
        stftx: complex spectra of input signal
    """
    w = 0.5*(1-np.cos(2*np.pi/fftsize*np.arange(fftsize)))
    awin = np.sqrt(w*2.0*stftshift/fftsize)
    T = numFrames(len(x), stftshift)
    F = fftsize//2+1
    stftx = np.zeros([F, T], dtype=np.complex)
    idx1 = 0
    for tau in range(T):
        tx = np.zeros(fftsize)
        idx2 = min(idx1+fftsize, len(x))
        tx[:idx2-idx1] = x[idx1:idx2]
        tx = tx*awin
        fx = np.fft.rfft(tx, fftsize)
        stftx[:, tau] = fx
        idx1 = idx1+stftshift
    return stftx


def istft(stftx, stftshift):
    """inverse-Short time DFT.

    Args:
        stftx (np.ndarray): complex spectra.
        stftshift (int): time shift.

    Returns:
        x: reconstruction of input spectra.
    """
    F, T = stftx.shape
    fftsize = (F-1)*2
    w = 0.5*(1-np.cos(2*np.pi/fftsize*np.arange(fftsize)))
    swin = np.sqrt(w*2.0*stftshift/fftsize)
    lenx = dataLength(T, stftshift, fftsize)
    x = np.zeros(lenx)
    idx1 = 0
    fx = np.zeros(fftsize, dtype=np.complex)
    for tau in range(T):
        fx[:F] = stftx[:, tau]
        fx[F-1:] = np.conj(stftx[1:, tau][::-1])
        tx = np.fft.ifft(fx, fftsize)
        tx = tx*swin
        tx = np.real(tx)
        x[idx1:idx1+fftsize] = x[idx1: idx1+fftsize]+tx
        idx1 = idx1+stftshift
    return x


if __name__ == "__main__":
    import soundfile as sf
    sig, sr = sf.read("sample.wav")
    stftx = stft(sig[:, 0], 502, 1024)
    output = istft(stftx, 502)
    output = output[:len(sig)]
    sf.write("reconstruct.wav", output, sr)
