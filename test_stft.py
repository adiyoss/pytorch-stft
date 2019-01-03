import librosa
import traceback
from stft import STFT
import torch

def test_stft():
    y, sr = librosa.load(librosa.util.example_audio_file())
    audio = torch.autograd.Variable(torch.FloatTensor(y), requires_grad=False).unsqueeze(0)

    if torch.cuda.is_available():
        audio = audio.cuda()

    def mse(ground_truth, estimated):
        return torch.mean((ground_truth - estimated)**2)

    def to_np(tensor):
        return tensor.cpu().data.numpy()

    filter_length=320
    hop_length=160
    stft = STFT(filter_length=filter_length, hop_length=hop_length)
    if torch.cuda.is_available():
        stft = stft.cuda()
    output, m_hat, p_hat = stft(audio)

    D = librosa.stft(y, n_fft=filter_length, hop_length=hop_length)
    m, p = librosa.magphase(D)

    import ipdb; ipdb.set_trace()

    import matplotlib.pyplot as plt
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='time')
    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    #for i in range(12):
    #    filter_length = 2**i
    #    for j in range(i+1):
    #        try:
    #            hop_length = 2**j
    #            stft = STFT(filter_length=filter_length, hop_length=hop_length)
    #            if torch.cuda.is_available():
    #                stft = stft.cuda()
    #            output, m_hat, p_hat = stft(audio)
    #            loss = mse(audio, output)
    #            print('Audio MSE: %s @ filter_length = %d, hop_length = %d' % (str(to_np(loss).item()), filter_length, hop_length))
    #        except:
    #            print('Failed @ filter_length = %d, hop_length = %d' % (filter_length, hop_length))
    #            print(traceback.print_exception())

if __name__ == "__main__":
    test_stft()
