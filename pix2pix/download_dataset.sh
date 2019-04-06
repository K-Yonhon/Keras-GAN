mkdir -p datasets
URL=https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/
wget -N -r -l 1 -w 1 -nd -nH -np -erobots=off -A tar.gz $URL -P ./datasets/
for FILE in `find $1 -name "*.tar.gz"`; do
tar -zxvf $FILE -C ./datasets/
rm $FILE
done

<< COMMENTOUT
mkdir datasets
FILE=$1
URL=https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/$FILE.tar.gz
TAR_FILE=./datasets/$FILE.tar.gz
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $TAR_FILE
mkdir $TARGET_DIR
tar -zxvf $TAR_FILE -C ./datasets/
rm $TAR_FILE
COMMENTOUT