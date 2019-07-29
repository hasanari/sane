for folder in *; do
  mkdir -p ../all_data/${folder%%}_sync/bin_data

  for file in ${folder%%}/*.bin; do
  	  target=${file/\//};
	  mv  ${file%%} ../all_data/${folder%%}_sync/bin_data/${target%%};

	done;
done



for file in *.txt; do
    y=${file%.*}
    target=${file/\//};
    echo ${y%%}

    mkdir -p ../all_data/${y%%}_sync/oxts


    mv  ${file%%} ../all_data/${y%%}_sync/oxts/${target%%};

done;


for file in *.txt; do
    y=${file%.*}
    target=${file/\//};
    echo ${y%%}

    mkdir -p ../all_data/${y%%}_sync/label


    mv  ${file%%} ../all_data/${y%%}_sync/label/${target%%};

done;



for folder in *; do
  mkdir -p ../../all_data/training_${folder%%}_sync/bin_data

  for file in ${folder%%}/*.bin; do
  	  target=${file/\//};
	  mv  ${file%%} ../../all_data/training_${folder%%}_sync/bin_data/${target%%};

	done;
done
