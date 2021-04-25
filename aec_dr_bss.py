# shmzhang@aslp, 2020-04
import numpy as np
import config as cfg
from heig2 import heig2
from stft import istft, stft


def aec_dr_bss(num_mics, num_refs, datain):
    """Perform aec, dr, and bss

    Args:
        num_mics (int): no. of mic channels
        num_refs (int): no. of reference channels
        datain (np.ndarray): input data

    Returns:
        dataout: output data
    """
    M = num_mics
    R = num_refs
    N = M
    Xtf = []
    for m in range(M + R):
        Xtf.append(stft(datain[m], cfg.stftshift, cfg.fftsize))
    K, T = Xtf[0].shape
    Ytf = []
    for m in range(M):
        Ytf.append(np.zeros([K, T], dtype=np.complex))
    # space for dr
    Micbuffer = np.zeros([K, M*(cfg.DR_DELAY+1)], dtype=np.complex)
    drfsize = M*cfg.DR_FLEN

    BufDR = np.zeros([K, drfsize], dtype=np.complex)
    Cmd, Cdd, Reverbpath = [], [], []
    for k in range(K):
        Cmd.append(np.zeros([M, drfsize], dtype=np.complex))
        Reverbpath.append(np.zeros([M, drfsize], dtype=np.complex))
        Cdd.append(np.zeros([drfsize, drfsize], dtype=np.complex))
    # space for aec
    aecfsize = R*cfg.AEC_FLEN
    BufAEC = np.zeros([K, aecfsize], dtype=np.complex)
    Cmr, Crr, Echopath = [], [], []
    for k in range(K):
        Cmr.append(np.zeros([M, aecfsize], dtype=np.complex))
        Crr.append(np.zeros([aecfsize, aecfsize], dtype=np.complex))
        Echopath.append(np.zeros([M, aecfsize], dtype=np.complex))

    # space for bss
    C1, C2, Demix = [], [], []
    for k in range(K):
        C1.append(cfg.STABLE_EPS*np.eye(M, M))
        C2.append(cfg.STABLE_EPS*np.eye(M, M))
        Demix.append(np.eye(N, M).astype(np.complex))
    # perform iteration
    for t in range(T):
        # perform aec.
        Nearend = np.zeros([K, M], dtype=np.complex)
        # shift in ref data
        # BufAEC=circshift(BufAEC, R, 2);
        BufAEC = np.roll(BufAEC, R, axis=1)
        for r in range(R):
            BufAEC[:, r] = Xtf[M+r][:, t]
        for k in range(K):
            mic = np.zeros([M, 1], dtype=np.complex)
            for m in range(M):
                mic[m] = Xtf[m][k, t]

            # ref=permute(BufAEC(k, :), [2, 1])
            ref = np.expand_dims(BufAEC[k, :], axis=1)

            echo = np.dot(Echopath[k], ref)

            nearend = mic-echo
            Nearend[k, :] = nearend[:, 0]

            xsq = np.abs(mic)**2
            ysq = np.abs(nearend)**2

            phi = sum(ysq[np.where(ysq < xsq)]) + \
                sum(xsq[np.where(ysq >= xsq)])
            # phi=(1-DRAEC_FORGET)*(phi+VAR_BIAS)^((GAMMA-2)/2)
            phi = (1-cfg.DRAEC_FORGET)*(phi+cfg.VAR_BIAS) ** ((cfg.GAMMA-2)/2)

            # update mic ref correlation
            Cmr[k] = cfg.DRAEC_FORGET*Cmr[k]+phi*(np.dot(mic, np.conj(ref).T))

            # update ref auto-correlation
            Crr[k] = cfg.DRAEC_FORGET*Crr[k]+phi*(np.dot(ref, np.conj(ref).T))

            # update echo path
            Echopath[k] = np.dot(Cmr[k], np.linalg.inv(
                Crr[k] + cfg.DRAEC_DIAGLOAD * np.eye(aecfsize, aecfsize)))
        # perform dr
        # direct and early reverberation
        Early = np.zeros([K, M], dtype=np.complex)

        #
        # shift in new data
        #
        # Micbuffer=circshift(Micbuffer, M, 2)
        Micbuffer = np.roll(Micbuffer, M, axis=1)
        Micbuffer[:, :M] = Nearend

        # BufDR=circshift(BufDR, M, 2)
        BufDR = np.roll(BufDR, M, axis=1)

        # BufDR(:, 1:M)=Micbuffer(:, end-M+1:end)
        BufDR[:, :M] = Micbuffer[:, -M:]

        for k in range(K):
            # mic=permute(Micbuffer(k, 1:M), [2,1])
            mic = np.expand_dims(Micbuffer[k, :M], axis=1)

            # ref=permute(BufDR(k, :), [2,1])
            ref = np.expand_dims(BufDR[k, :], axis=1)

            # calculate late reverberation
            late = np.dot(Reverbpath[k], ref)

            # direct and early reverberation
            early = mic-late

            Early[k, :] = early[:, 0]

            # calculate nonlinearity
            xsq = np.abs(mic)**2
            ysq = np.abs(early)**2
            phi = sum(ysq[np.where(ysq < xsq)]) + \
                sum(xsq[np.where(ysq >= xsq)])

            phi = (1-cfg.DRAEC_FORGET)*(phi+cfg.VAR_BIAS)**((cfg.GAMMA-2)/2)

            # update mic ref correlation
            Cmd[k] = cfg.DRAEC_FORGET*Cmd[k] + \
                np.dot(phi, np.dot(mic, np.conj(ref).T))

            # update ref auto-correlation
            Cdd[k] = cfg.DRAEC_FORGET*Cdd[k] + \
                np.dot(phi, np.dot(ref, np.conj(ref).T))

            # update reverb path
            Reverbpath[k] = np.dot(Cmd[k], np.linalg.inv(
                Cdd[k]+cfg.DRAEC_DIAGLOAD*np.eye(drfsize, drfsize)))

        # perform bss
        Bssout = np.zeros([K, M], dtype=np.complex)

        # calculate nonlinearity
        phi1 = 0
        phi2 = 0
        for k in range(K):
            x = Early[k, :].T

            y = np.dot(Demix[k], x)
            Bssout[k, :] = y.T
            phi1 = phi1+np.abs(y[0])**2
            phi2 = phi2+np.abs(y[1])**2

        phi1 = (1-cfg.BF_FORGET)*(phi1+cfg.VAR_BIAS)**((cfg.GAMMA-2)/2)
        phi2 = (1-cfg.BF_FORGET)*(phi2+cfg.VAR_BIAS)**((cfg.GAMMA-2)/2)
        # update the demixing matrices
        for k in range(K):
            # accumulate the weighted correlation
            x = Early[k, :].reshape(-1, 1)
            C1[k] = cfg.BF_FORGET*C1[k]+phi1 * np.dot(x, np.conj(x).T)
            C2[k] = cfg.BF_FORGET*C2[k]+phi2 * np.dot(x, np.conj(x).T)
            # solve gev problem
            D = heig2(cfg.BF_DIAGLOAD, C2[k], C1[k])
            Demix[k] = D
        for m in range(M):
            Ytf[m][:, t] = Bssout[:, m]
    # perform istft and output signal
    dataout = []
    for n in range(N):
        dataout.append(istft(Ytf[n], cfg.stftshift))
    return dataout


if __name__ == "__main__":
    import soundfile as sf
    # Nearend signal, equal to nummics.
    N = 2
    # sensor numbers.
    nummics = 2
    # references signal.
    numrefs = 1
    testdata, sr = sf.read("sample.wav")
    # [M0, M1, R0] stack.
    testdata = [testdata[:, i] for i in range(nummics + numrefs)]
    lenx = len(testdata[0])
    # [N0, N1]
    output = aec_dr_bss(nummics, numrefs, testdata)
    # stft clips.
    for i in range(N):
        output[i] = output[i][:lenx]
    sf.write("output_0.wav", output[0], sr)
    sf.write("output_1.wav", output[1], sr)
