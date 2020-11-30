YAHOO_DS="../../Stochastic-Methods/data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/"
DATA_SOURCE_DIR="A4Benchmark"
PATTERN="*TS*.csv"
#########################################################
#'A1Benchmark' : "*.csv" 
#'A2Benchmark' : "*.csv"
#'A3Benchmark' : "*TS*.csv"
#'A4Benchmark' : "*TS*.csv" 

############################################################
DATA_FOLDER=$YAHOO_DS$DATA_SOURCE_DIR"/"
echo "DATA_FOLDER : $DATA_FOLDER"
for fl in `ls $DATA_FOLDER$PATTERN`
do
    #echo $fl
    FL_NAME=`echo $fl | rev |cut -d'/' -f1  | rev`
    #echo $FL_NAME
    CMD="python Yahoo-LSTM.py -f $DATA_SOURCE_DIR --save 1 --file_name $FL_NAME --num_threads_inter 2 --epochs 20 --n_iter 3 --patience 3"
    echo "Staring command  : $CMD  at time :: `date`"
    start_timestamp=$(date +%s)
    $CMD
    end_timestamp=$(date +%s)
    run_time=`expr $end_timestamp - $start_timestamp`
    echo "Finished command : $CMD :: took $run_time seconds"
    sleep 5
done
