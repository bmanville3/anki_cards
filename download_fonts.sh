mkdir -p ./fonts

fonts=(
  "ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf"
  "ofl/notoserifjp/NotoSerifJP%5Bwght%5D.ttf"
  "ofl/mplusrounded1c/MPLUSRounded1c-Regular.ttf"
  "ofl/mplus1p/MPLUS1p-Regular.ttf"
  "apache/kosugimaru/KosugiMaru-Regular.ttf"
  "ofl/sawarabigothic/SawarabiGothic-Regular.ttf"
  "ofl/sawarabimincho/SawarabiMincho-Regular.ttf"
  "ofl/zenkakugothicnew/ZenKakuGothicNew-Regular.ttf"
  "ofl/zenantique/ZenAntique-Regular.ttf"
  "ofl/shipporimincho/ShipporiMincho-Regular.ttf"
)

for font in "${fonts[@]}"; do
    name=$(basename "$font")

    if [ ! -f "./fonts/$name" ]; then
        echo "Downloading $name"

        curl -g -L --fail \
            -o "./fonts/$name" \
            "https://raw.githubusercontent.com/google/fonts/main/$font"
    else
        echo "$name already exists, skipping."
    fi
done