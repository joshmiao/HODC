mkdir ETH3D/two_view_testing -p
cd ETH3D/two_view_testing
wget https://www.eth3d.net/data/two_view_test.7z
7za x two_view_test.7z
rm -vf two_view_test.7z
cd ../../

mkdir ETH3D/two_view_training -p
cd ETH3D/two_view_training
wget https://www.eth3d.net/data/two_view_training.7z
7za x two_view_training.7z
rm -vf two_view_training.7z
wget https://www.eth3d.net/data/two_view_training_gt.7z
7za x two_view_training_gt.7z
rm -vf two_view_training_gt.7z
cd ../../

