mkdir Middlebury -p
cd Middlebury/
# wget https://www.dropbox.com/s/fn8siy5muak3of3/official_train.txt -P MiddEval3/
wget https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-Q.zip
unzip MiddEval3-data-Q.zip
wget https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-Q.zip
unzip MiddEval3-GT0-Q.zip
wget https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-H.zip
unzip MiddEval3-data-H.zip
wget https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-H.zip
unzip MiddEval3-GT0-H.zip
wget https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-F.zip
unzip MiddEval3-data-F.zip
wget https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-F.zip
unzip MiddEval3-GT0-F.zip
rm *.zip
cd ../