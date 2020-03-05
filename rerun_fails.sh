pizza=$(grep generic3 -c  *-6131*.o | grep :1 )
echo uuu $pizza
for zzz in $pizza
do
  #www=$(sed ':1' '' $zzz)
  ccc=$(grep generic3  ${zzz//":1"/""})
cat iter.job <(echo $ccc )  > nana.job

bsub  -R "select[mem>4096] rusage[mem=4000]" <nana.job

done
