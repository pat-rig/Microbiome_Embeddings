glove_dir="/home/koehler/GloVe-master/build"
filter=standard

echo "filter"
echo "$filter"

echo "vocab count"
$glove_dir/vocab_count -verbose 2 -max-vocab 50000 -min-count 10 < glove_input_file > vocab_filter"$filter".txt

echo "Coooccurence count"
$glove_dir/cooccur -verbose 2 -symmetric 1 -window-size 50000 -vocab-file vocab_filter"$filter".txt -memory 20 -overflow-file overflow_filter"$filter" < glove_input_file > cooccur_filter"$filter".bin

echo "Cooccurent shuffle"
$glove_dir/shuffle -verbose 2 -memory 20 < cooccur_filter"$filter".bin  > cooccur_filter"$filter".shuf.bin

echo "Run Glove 100"
$glove_dir/glove -input-file cooccur_filter"$filter".shuf.bin -vocab-file vocab_filter"$filter".txt -save-file glove_emb_filter -gradsq-file gradsq_filter"$filter" -verbose 2 -vector-size 100 -threads 96  -iter 100

echo "DONE"
