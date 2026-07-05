for d in *long/*/; do (cd $d && \
    rm -f centered.gro\

    echo -e "18\n0\n" | gmx trjconv -f npt_long.gro -s *.tpr -o centered.gro -center -pbc mol && \
    echo -e "16\n0\n" | gmx trjconv -f centered.gro -s *.tpr -o centered2.gro -center -pbc mol && \

    halfz1=$(tail -1 centered.gro  | awk '{print $3/2}') && \
    halfz2=$(tail -1 centered2.gro | awk '{print $3/2}') && \

    d1=$(awk -v h=$halfz1 'NR==2{n=$1} NR>2 && NR<=n+2 {name=substr($0,11,5); gsub(/ /,"",name); if(name~/^C214/){z=(NF>=8)?$(NF-3):$NF; dz=z-h; if(dz<0)dz=-dz; ds+=dz; zn++}} END{md=ds/zn; printf "centered.gro:  mean |z - half_z| = %.3f  (n=%d)\n", md, zn > "/dev/stderr"; print md}' centered.gro) && \
    d2=$(awk -v h=$halfz2 'NR==2{n=$1} NR>2 && NR<=n+2 {name=substr($0,11,5); gsub(/ /,"",name); if(name~/^C214/){z=(NF>=8)?$(NF-3):$NF; dz=z-h; if(dz<0)dz=-dz; ds+=dz; zn++}} END{md=ds/zn; printf "centered2.gro: mean |z - half_z| = %.3f  (n=%d)\n", md, zn > "/dev/stderr"; print md}' centered2.gro) && \

    if awk -v a=$d1 -v b=$d2 'BEGIN{exit !(a<b)}'; then \
        rm centered2.gro; \
    else \
        rm centered.gro && mv centered2.gro centered.gro; \
    fi \
); done
