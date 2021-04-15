for I in 1
do
  rm -rf mass_spring/final5
  python3 mass_spring_interactive.py 5
  cd mass_spring/final5
  ti video && ti gif -i video.mp4 -f250 && mv video.gif ../$I.gif
  cd ../..
done
