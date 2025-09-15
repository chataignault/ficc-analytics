#!/bin/bash

echo "Copy train and test House Price data to the current folder"

HOUSE_PRICES_COMPETITION_FOLDER="../../kaggle/house_prices"

if [ ! -d "$HOUSE_PRICES_COMPETITION_FOLDER" ]; then
	echo "House Prices competition is not loaded in the kaggle folder"
	return
fi

if [ ! -f "$HOUSE_PRICES_COMPETITION_FOLDER/train.csv" ]; then 
	echo "Missing train.csv"
	return
fi

if [ ! -f "$HOUSE_PRICES_COMPETITION_FOLDER/test.csv" ]; then 
	echo "Missing test.csv"
	return
fi

cp "$HOUSE_PRICES_COMPETITION_FOLDER/train.csv" . 
cp "$HOUSE_PRICES_COMPETITION_FOLDER/test.csv" . 

echo "House prices data copied successfully in the current folder."

