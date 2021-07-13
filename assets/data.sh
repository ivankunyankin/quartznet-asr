 echo "Downloading LibriTTS data"
        mkdir -p LibriTTS
        curl https://www.openslr.org/resources/60/train-clean-360.tar.gz  --output LibriTTS/train-clean-360.tar.gz
        curl https://www.openslr.org/resources/60/dev-clean.tar.gz  --output LibriTTS/dev-clean.tar.gz
        curl https://www.openslr.org/resources/60/test-clean.tar.gz --output LibriTTS/test-clean.tar.gz
        echo "Unzipping..."
        pv LibriTTS/train-clean-360.tar.gz  | tar xzf - -C LibriTTS/
        pv LibriTTS/dev-clean.tar.gz  | tar xzf - -C LibriTTS/
        pv LibriTTS/test-clean.tar.gz | tar xzf - -C LibriTTS/