# final_analysis_mic.tcl
#
# Usage:
# vmd -dispdev text -e final_analysis_mic.tcl -args input.gro lipidSel headSel tailSel phosphorSel peptideSel nPeptides output.csv cutoffNm lambdaLipidNm lambdaPeptideNm
#
# Example:
# vmd -dispdev text -e final_analysis_mic.tcl -args "system.gro" "resname POPC POPS" "name P" "name C218 C316" "name P" "resname ILE ASN TRP LYN LYS GLY MET ALA AL1 AL2 LEU NHE mol ME1 ME2" 8 "out.csv" 20.0 20.0 10.0

proc vecdot {a b} {
    expr {[lindex $a 0]*[lindex $b 0] + [lindex $a 1]*[lindex $b 1] + [lindex $a 2]*[lindex $b 2]}
}

proc vecnorm {v} {
    expr {sqrt([vecdot $v $v])}
}

proc vecsub {a b} {
    list \
        [expr {[lindex $a 0]-[lindex $b 0]}] \
        [expr {[lindex $a 1]-[lindex $b 1]}] \
        [expr {[lindex $a 2]-[lindex $b 2]}]
}

proc clamp {x lo hi} {
    if {$x < $lo} { return $lo }
    if {$x > $hi} { return $hi }
    return $x
}

proc angle_vs_z_deg {v} {
    set z {0.0 0.0 1.0}
    set nv [vecnorm $v]
    if {$nv < 1e-12} { return -1 }
    set c [expr {[vecdot $v $z]/$nv}]
    set c [clamp $c -1.0 1.0]
    expr {acos($c)*180.0/acos(-1.0)}
}

proc mat3_det {M} {
    set a [lindex $M 0 0]; set b [lindex $M 0 1]; set c [lindex $M 0 2]
    set d [lindex $M 1 0]; set e [lindex $M 1 1]; set f [lindex $M 1 2]
    set g [lindex $M 2 0]; set h [lindex $M 2 1]; set i [lindex $M 2 2]
    expr {$a*($e*$i - $f*$h) - $b*($d*$i - $f*$g) + $c*($d*$h - $e*$g)}
}

proc mat3_inv {M} {
    set a [lindex $M 0 0]; set b [lindex $M 0 1]; set c [lindex $M 0 2]
    set d [lindex $M 1 0]; set e [lindex $M 1 1]; set f [lindex $M 1 2]
    set g [lindex $M 2 0]; set h [lindex $M 2 1]; set i [lindex $M 2 2]

    set det [mat3_det $M]
    if {abs($det) < 1e-14} {
        error "Cell matrix is singular; cannot apply triclinic MIC."
    }

    set A [expr {( $e*$i - $f*$h) / $det}]
    set B [expr {( $c*$h - $b*$i) / $det}]
    set C [expr {( $b*$f - $c*$e) / $det}]

    set D [expr {( $f*$g - $d*$i) / $det}]
    set E [expr {( $a*$i - $c*$g) / $det}]
    set F [expr {( $c*$d - $a*$f) / $det}]

    set G [expr {( $d*$h - $e*$g) / $det}]
    set H [expr {( $b*$g - $a*$h) / $det}]
    set I [expr {( $a*$e - $b*$d) / $det}]

    list [list $A $B $C] [list $D $E $F] [list $G $H $I]
}

proc mat3_vec {M v} {
    list \
        [expr {[lindex $M 0 0]*[lindex $v 0] + [lindex $M 0 1]*[lindex $v 1] + [lindex $M 0 2]*[lindex $v 2]}] \
        [expr {[lindex $M 1 0]*[lindex $v 0] + [lindex $M 1 1]*[lindex $v 1] + [lindex $M 1 2]*[lindex $v 2]}] \
        [expr {[lindex $M 2 0]*[lindex $v 0] + [lindex $M 2 1]*[lindex $v 1] + [lindex $M 2 2]*[lindex $v 2]}]
}

proc deg2rad {x} {
    expr {$x * acos(-1.0) / 180.0}
}

proc build_cell_matrices {a b c alpha beta gamma} {
    set ar [deg2rad $alpha]
    set br [deg2rad $beta]
    set gr [deg2rad $gamma]

    set ax $a
    set ay 0.0
    set az 0.0

    set bx [expr {$b * cos($gr)}]
    set by [expr {$b * sin($gr)}]
    set bz 0.0

    set cx [expr {$c * cos($br)}]
    if {abs(sin($gr)) < 1e-12} {
        error "Invalid gamma angle."
    }
    set cy [expr {$c * (cos($ar) - cos($br)*cos($gr)) / sin($gr)}]

    set cz2 [expr {$c*$c - $cx*$cx - $cy*$cy}]
    if {$cz2 < 0 && $cz2 > -1e-10} { set cz2 0.0 }
    if {$cz2 < 0} { error "Invalid cell geometry (cz^2 < 0)." }
    set cz [expr {sqrt($cz2)}]

    set H [list [list $ax $ay $az] [list $bx $by $bz] [list $cx $cy $cz]]
    set Hinv [mat3_inv $H]
    return [list $H $Hinv]
}

proc get_frame_cell {mol f} {
    molinfo $mol set frame $f
    set a [molinfo $mol get a]
    set b [molinfo $mol get b]
    set c [molinfo $mol get c]
    set alpha [molinfo $mol get alpha]
    set beta  [molinfo $mol get beta]
    set gamma [molinfo $mol get gamma]

    if {$a <= 0 || $b <= 0 || $c <= 0} {
        error "Invalid cell lengths at frame $f: a=$a b=$b c=$c"
    }

    build_cell_matrices $a $b $c $alpha $beta $gamma
}

proc mic_delta {d H Hinv} {
    set s [mat3_vec $Hinv $d]
    set sw [list \
        [expr {[lindex $s 0] - round([lindex $s 0])}] \
        [expr {[lindex $s 1] - round([lindex $s 1])}] \
        [expr {[lindex $s 2] - round([lindex $s 2])}] \
    ]
    mat3_vec $H $sw
}

proc pbc_dist {a b H Hinv} {
    set d [vecsub $a $b]
    set dm [mic_delta $d $H $Hinv]
    vecnorm $dm
}

proc build_peptide_index_groups {mol peptideSelText nPeptides} {
    if {$nPeptides < 1} { error "nPeptides must be >= 1" }

    set psel [atomselect $mol $peptideSelText frame 0]
    if {[$psel num] < 1} {
        $psel delete
        error "No atoms matched peptide selection: $peptideSelText"
    }

    set idxs [$psel get index]
    $psel delete

    set idxs [lsort -integer $idxs]
    set natoms [llength $idxs]

    if {$natoms < $nPeptides} {
        error "Peptide selection has only $natoms atoms, fewer than nPeptides=$nPeptides"
    }
    if {[expr {$natoms % $nPeptides}] != 0} {
        error "Peptide atom count ($natoms) is not divisible by nPeptides ($nPeptides)"
    }

    set atomsPerPeptide [expr {$natoms / $nPeptides}]
    set groups {}
    for {set p 0} {$p < $nPeptides} {incr p} {
        set s [expr {$p * $atomsPerPeptide}]
        set e [expr {$s + $atomsPerPeptide - 1}]
        lappend groups [lrange $idxs $s $e]
    }

    return [list $groups $atomsPerPeptide]
}

proc point_to_peptide_metrics {point pepCOMs cutoff lambdaLipid nPeptides H Hinv} {
    set minD 1e30
    set sumD 0.0
    set cntD 0
    set nWithin 0
    set wsum 0.0

    foreach pc $pepCOMs {
        if {$pc eq ""} { continue }
        set d [pbc_dist $point $pc $H $Hinv]

        if {$d < $minD} { set minD $d }
        set sumD [expr {$sumD + $d}]
        incr cntD

        if {$d <= $cutoff} { incr nWithin }
        set wsum [expr {$wsum + exp(-$d / $lambdaLipid)}]
    }

    if {$cntD > 0} {
        set meanD [expr {$sumD / double($cntD)}]
        set fracWithin [expr {$nWithin / double($nPeptides)}]
    } else {
        set minD ""
        set meanD ""
        set nWithin ""
        set fracWithin ""
        set wsum ""
    }

    return [list $minD $meanD $nWithin $fracWithin $wsum]
}

# Returns:
# cluster_raw, cluster_norm, mean_pair_dist, mean_pair_weight, valid_peptides, pair_count
proc peptide_cluster_metrics {pepCOMs H Hinv lambdaPep nPeptides} {
    if {$lambdaPep <= 0} { error "lambdaPeptide must be > 0" }

    set validIdx {}
    for {set i 0} {$i < $nPeptides} {incr i} {
        if {[lindex $pepCOMs $i] ne ""} {
            lappend validIdx $i
        }
    }

    set nValid [llength $validIdx]
    if {$nValid < 2} {
        return [list "" "" "" "" $nValid 0]
    }

    array set ci {}
    foreach i $validIdx {
        set ci($i) 0.0
    }

    set sumPairDist 0.0
    set sumPairWeight 0.0
    set nPairs 0

    for {set a 0} {$a < $nValid} {incr a} {
        set i [lindex $validIdx $a]
        set pi [lindex $pepCOMs $i]

        for {set b [expr {$a + 1}]} {$b < $nValid} {incr b} {
            set j [lindex $validIdx $b]
            set pj [lindex $pepCOMs $j]

            set d [pbc_dist $pi $pj $H $Hinv]
            set w [expr {exp(-$d / $lambdaPep)}]

            set sumPairDist [expr {$sumPairDist + $d}]
            set sumPairWeight [expr {$sumPairWeight + $w}]
            incr nPairs

            set ci($i) [expr {$ci($i) + $w}]
            set ci($j) [expr {$ci($j) + $w}]
        }
    }

    set sumCi 0.0
    foreach i $validIdx {
        set sumCi [expr {$sumCi + $ci($i)}]
    }

    set clusterRaw [expr {$sumCi / double($nValid)}]
    set clusterNorm [expr {$clusterRaw / double($nValid - 1)}]
    set meanPairDist [expr {$sumPairDist / double($nPairs)}]
    set meanPairWeight [expr {$sumPairWeight / double($nPairs)}]

    return [list $clusterRaw $clusterNorm $meanPairDist $meanPairWeight $nValid $nPairs]
}

proc lipid_tilt_and_multipeptide_metrics {lipidSelText headSelText tailSelText phosphorSelText peptideSelText nPeptides outFile cutoff lambdaLipid lambdaPep} {
    set mol [molinfo top]
    if {$mol eq "" || $mol < 0} {
        error "No molecule loaded."
    }
    if {$lambdaLipid <= 0} { error "lambdaLipid must be > 0" }
    if {$lambdaPep <= 0} { error "lambdaPeptide must be > 0" }

    set nframes [molinfo $mol get numframes]

    set allLip [atomselect $mol $lipidSelText frame 0]
    if {[$allLip num] < 1} {
        $allLip delete
        error "No atoms matched lipid selection: $lipidSelText"
    }

    set residList   [$allLip get resid]
    set segidList   [$allLip get segid]
    set resnameList [$allLip get resname]
    $allLip delete

    array set seenLip {}
    set lipids {}
    set n [llength $residList]
    for {set i 0} {$i < $n} {incr i} {
        set segid   [lindex $segidList $i]
        set resid   [lindex $residList $i]
        set resname [lindex $resnameList $i]
        set key "${segid}:${resid}:${resname}"
        if {![info exists seenLip($key)]} {
            set seenLip($key) 1
            lappend lipids [list $segid $resid $resname]
        }
    }

    lassign [build_peptide_index_groups $mol $peptideSelText $nPeptides] pepIndexGroups atomsPerPeptide

    puts "Found [llength $lipids] lipids"
    puts "Peptides: $nPeptides"
    puts "Atoms per peptide: $atomsPerPeptide"
    puts "Processing $nframes frame(s)..."

    set fh [open $outFile w]
    puts $fh "frame,segid,resid,resname,angle_deg,angle_folded_0_90_deg,lipid_com_x,lipid_com_y,lipid_com_z,P_x,P_y,P_z,min_dist_COM_to_peptide_com,mean_dist_COM_to_all_peptide_coms,n_within_cutoff_COM,fraction_within_cutoff_COM,weighted_proximity_sum_COM,min_dist_P_to_peptide_com,mean_dist_P_to_all_peptide_coms,n_within_cutoff_P,fraction_within_cutoff_P,weighted_proximity_sum_P,peptide_cluster_score_frame,peptide_cluster_score_frame_norm,peptide_mean_pair_dist,peptide_mean_pair_weight,peptide_valid_count,peptide_pair_count"

    for {set f 0} {$f < $nframes} {incr f} {
        lassign [get_frame_cell $mol $f] H Hinv

        set pepCOMs {}
        for {set p 0} {$p < $nPeptides} {incr p} {
            set idxChunk [lindex $pepIndexGroups $p]
            set idxStr [join $idxChunk " "]
            set psel [atomselect $mol "index $idxStr" frame $f]
            if {[$psel num] < 1} {
                lappend pepCOMs ""
            } else {
                lappend pepCOMs [measure center $psel weight mass]
            }
            $psel delete
        }

        lassign [peptide_cluster_metrics $pepCOMs $H $Hinv $lambdaPep $nPeptides] \
            pepClusterRaw pepClusterNorm pepMeanPairDist pepMeanPairWeight pepValidCount pepPairCount

        foreach lip $lipids {
            lassign $lip segid resid resname
            set baseSel "segid \"$segid\" and resid $resid and resname $resname"

            set headSel [atomselect $mol "($baseSel) and ($headSelText)" frame $f]
            set tailSel [atomselect $mol "($baseSel) and ($tailSelText)" frame $f]
            set pSel    [atomselect $mol "($baseSel) and ($phosphorSelText)" frame $f]
            set lipSel  [atomselect $mol "($baseSel)" frame $f]

            if {[$headSel num] < 1 || [$tailSel num] < 1 || [$pSel num] < 1 || [$lipSel num] < 1} {
                $headSel delete
                $tailSel delete
                $pSel delete
                $lipSel delete
                continue
            }

            set headCOM [measure center $headSel weight mass]
            set tailCOM [measure center $tailSel weight mass]
            set pPos    [measure center $pSel weight mass]
            set lipCOM  [measure center $lipSel weight mass]

            set v [mic_delta [vecsub $tailCOM $headCOM] $H $Hinv]
            set ang [angle_vs_z_deg $v]
            if {$ang < 0} {
                set folded ""
            } else {
                set folded [expr {$ang > 90.0 ? 180.0 - $ang : $ang}]
            }

            lassign [point_to_peptide_metrics $lipCOM $pepCOMs $cutoff $lambdaLipid $nPeptides $H $Hinv] \
                minCOM meanCOM nWithinCOM fracCOM wsumCOM

            lassign [point_to_peptide_metrics $pPos $pepCOMs $cutoff $lambdaLipid $nPeptides $H $Hinv] \
                minP meanP nWithinP fracP wsumP

            puts $fh "$f,$segid,$resid,$resname,$ang,$folded,[lindex $lipCOM 0],[lindex $lipCOM 1],[lindex $lipCOM 2],[lindex $pPos 0],[lindex $pPos 1],[lindex $pPos 2],$minCOM,$meanCOM,$nWithinCOM,$fracCOM,$wsumCOM,$minP,$meanP,$nWithinP,$fracP,$wsumP,$pepClusterRaw,$pepClusterNorm,$pepMeanPairDist,$pepMeanPairWeight,$pepValidCount,$pepPairCount"

            $headSel delete
            $tailSel delete
            $pSel delete
            $lipSel delete
        }

        if {($f % 10) == 0} {
            puts "  frame $f / [expr {$nframes-1}]"
        }
    }

    close $fh
    puts "Done. Wrote: $outFile"
}

if {[llength $argv] != 11} {
    puts "Usage:"
    puts "vmd -dispdev text -e final_analysis_mic.tcl -args input.gro lipidSel headSel tailSel phosphorSel peptideSel nPeptides output.csv cutoffNm lambdaLipidNm lambdaPeptideNm"
    quit 1
}

set inGro        [lindex $argv 0]
set lipidSel     [lindex $argv 1]
set headSel      [lindex $argv 2]
set tailSel      [lindex $argv 3]
set phosphorSel  [lindex $argv 4]
set peptideSel   [lindex $argv 5]
set nPeptides    [lindex $argv 6]
set outCsv       [lindex $argv 7]
set cutoffNm     [lindex $argv 8]
set lambdaLipid  [lindex $argv 9]
set lambdaPep    [lindex $argv 10]

mol new $inGro type gro waitfor all

lipid_tilt_and_multipeptide_metrics \
    $lipidSel \
    $headSel \
    $tailSel \
    $phosphorSel \
    $peptideSel \
    $nPeptides \
    $outCsv \
    $cutoffNm \
    $lambdaLipid \
    $lambdaPep

quit 0
