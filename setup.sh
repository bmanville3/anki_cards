if [[ "$(basename "$PWD")" != "anki_cards" ]]; then
    echo "Wrong directory! Please run from 'anki_cards'"
    exit 1
fi

mkdir -p ./models/

if command -v ffmpeg >/dev/null 2>&1 ; then
    echo "FFmpeg is installed."
else
    echo "Installing FFmpeg..."
    if [ "$(uname)" == "Darwin" ]; then
        brew install ffmpeg
    else
        sudo apt install ffmpeg
    fi
fi

pip install -r requirements.txt

mkdir -p ./yomitan-jlpt-vocab
for i in 1 2 3 4 5; do
    curl -L --fail -o ./yomitan-jlpt-vocab/term_meta_bank_${i}.json \
      "https://raw.githubusercontent.com/stephenmk/yomitan-jlpt-vocab/main/yomitan-jlpt-vocab/term_meta_bank_${i}.json" \
      || echo "ERROR: failed to download term_meta_bank_${i}.json"
done

./download_fonts.sh
