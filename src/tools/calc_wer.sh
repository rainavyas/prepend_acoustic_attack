hyp_file=experiments/librispeech/whisper-tiny/predictions.json
ref_file=~/rds/rds-altaslp-8YSp2LXTlkY/data/librispeech/test_other/text

python3 src/tools/calc_wer.py $hyp_file $ref_file

hyp_process=${hyp_file}_hyp
ref_process=${hyp_file}_ref
~/rds/rds-altaslp-8YSp2LXTlkY/experiments/mq227/toolkits/SCTK-master/bin/sclite -r $ref_process trn -h $hyp_process trn -i rm -o dtl all stdout > ${hyp_file}.wer

grep Sum ${hyp_file}.wer
