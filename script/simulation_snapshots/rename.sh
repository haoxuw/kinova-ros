
cnt=0
for i in *.jpg;
do name=`echo "$i"`
   new_name=$(printf "%04d.jpg" "$cnt")
   mv $name $new_name
   cnt=$((cnt + 1))
   
   #ffmpeg -i "$i" "${name}.mov"
done
