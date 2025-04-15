#!/bin/bash

echo $TERM

printf "Basic 8 Colors (Text):\n"
for i in {30..37}; do printf "\033[%sm %s \033[0m" "$i" "Color($i)"; done; printf "\n"
printf "Basic 8 Colors (Background):\n"
for i in {40..47}; do printf "\033[%sm %s \033[0m" "$i" " Color($i) "; done; printf "\n\n"

printf "Bright/Bold 8 Colors (Text):\n"
for i in {90..97}; do printf "\033[%sm %s \033[0m" "$i" "Color($i)"; done; printf "\n"
printf "Bright/Bold 8 Colors (Background):\n"
for i in {100..107}; do printf "\033[%sm %s \033[0m" "$i" " Color($i) "; done; printf "\n\n"

printf "\nTesting 256 Colors:\n"
for i in {0..255}; do
    printf "\033[48;5;%sm%3d\033[0m " "$i" "$i"
    if (( (i + 1) % 16 == 0 )); then
        printf "\n"
    fi
done
printf "\n"

printf "\nTesting True Color (24-bit):\n"
printf "If the next line is a smooth gradient, True Color is likely supported:\n"
awk 'BEGIN{
    for (i=0; i<256; i++) {
        r=int(255*sin(i*3.14159/256));
        g=int(255*sin((i+85)*3.14159/256));
        b=int(255*sin((i+170)*3.14159/256));
        printf "\033[48;2;%d;%d;%dm \033[0m", r, g, b;
    }
    printf "\n";
}'