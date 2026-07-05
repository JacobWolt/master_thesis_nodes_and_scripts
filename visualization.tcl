# set c_atoms [atomselect top "name =~ \"C.*\""]
set n_atoms [atomselect top "name =~ \"N.*\""]
set o_atoms [atomselect top "name =~ \"O.*\""]
set h_atoms [atomselect top "name =~ \"H.*\""]
set cl_atoms [atomselect top "name =~ \"Cl.*\""]
set c1_atoms [atomselect top "resname AL1 AL2 mol ME1 ME2 LE1 LE2 LE3 LE4 RE1 RE2 and name =~ \"C.*\""]
set re_atoms [atomselect top "name =~ \"Re.*\""]
set ru_atoms [atomselect top "name =~ \"Ru.*\""]

$h_atoms set element "H"
# $c_atoms set element "C"
$n_atoms set element "N"
$o_atoms set element "O"
$c1_atoms set element "C"
# $f_atoms set element "F"
$ru_atoms set element "Ru"
$cl_atoms set element "Cl"
$re_atoms set element "Re"

material add MyMaterial
material change ambient MyMaterial 1.0
material change diffuse MyMaterial 0.0
material change specular MyMaterial 0.0
material change shininess MyMaterial 0.0
material change opacity MyMaterial 0.25
material change outline MyMaterial 0.0
material change outlinewidth MyMaterial 0.0

# Change element colors
color change rgb silver 0.6 0.6 0.6
color change rgb green2 0.8 1.0 0.65
color change rgb cyan3 0.7 0.9 1.0

color element H white
color Name H white
color Type H white

color element C gray
color name C gray
color Type C gray

color element Ru purple 
color Name Ru purple
color Type Ru purple

color element Cl cyan
color Name Cl cyan
color Type Cl cyan

# Display settings
display projection Orthographic
display depthcue off

color Display Background white
color Labels Bonds blue
color Labels Atoms black

label textsize 1.65
label textthickness 2.5

mol modselect 0 0 water 
mol color Name
mol representation Lines 1.000000
mol selection water 
mol material Transparent
mol modrep 0 0
mol addrep 0
mol modselect 1 0 resname ILE ASN TRP LYN LYS GLY MET ALA AL1 AL2 LEU NHE ACE AL1 AL2 mol ME1 ME2 LE1 LE2 LE3 LE4 RE1 RE2
mol modstyle 1 0 NewCartoon 0.300000 30.000000 4.100000 0
mol color ColorID 2
mol representation NewCartoon 0.300000 30.000000 4.100000 0
mol selection resname ILE ASN TRP LYN LYS GLY MET ALA AL1 AL2 LEU NHE ACE AL1 AL2 mol ME1 ME2 LE1 LE2 LE3 LE4 RE1 RE2
mol material AOShiny
mol modrep 1 0
mol addrep 0
mol modselect 2 0 resname AL1 AL2 mol ME1 ME2 LE1 LE2 LE3 LE4 RE1 RE2 
mol modstyle 2 0 Licorice 0.400000 30.000000 30.000000
mol color Type
mol representation Licorice 0.400000 30.000000 30.000000 
mol selection resname AL1 AL2 mol ME1 ME2 LE1 LE2 LE3 LE4 RE1 RE2
mol material AOShiny
mol modrep 2 0
mol addrep 0
mol modselect 3 0 resname CL
mol color ColorID 10
mol representation VDW 0.4 30.000000
mol selection resname CL
mol material AOShiny
mol modrep 3 0
mol addrep 0
mol modselect 4 0 resname NA
mol color ColorID 30
mol representation VDW 0.4 30.000000
mol selection resname NA
mol material AOShiny
mol modrep 4 0
mol addrep 0
mol modselect 5 0 resname DPPC POPC POPS and noh
mol color Name
mol representation lines 1.00000
mol selection resname DPPC POPC POPS and noh
mol material Transparent
mol modrep 5 0
mol addrep 0
mol modselect 6 0 element Cl
mol color ColorID 12 
mol representation licorice 0.410000 30.000000 30.000000
mol selection element Cl
mol material Opaque
mol modrep 6 0
mol addrep 0
mol modselect 7 0 element Re
mol color ColorID 3 
mol representation licorice 0.410000 30.000000 30.000000
mol selection element Re
mol material Opaque
mol modrep 7 0
mol addrep 0
mol modselect 8 0 element Ru
mol color ColorID 3
mol representation licorice 0.410000 30.000000 30.000000
mol selection element Ru
mol material Opaque
mol modrep 8 0

display projection Orthographic

display rendermode GLSL

display shadows on
display ambientocclusion on

display resize 2000 2000
scale by 1.9

rotate y by 90

rotate z by 90

# cd C:/Users/marti/Documents/Study/metallopeptide_thesis/BoCoFlow/\

# # Orientation and centering
# set _pep_sel_text "resname ILE ASN TRP LYN LYS GLY MET ALA AL1 AL2 LEU NHE ACE mol ME1 ME2 LE1 LE2 LE3 LE4 RE1 RE2"
# source tcl_scripts/orientation.tcl

# # Calculate immersion and peptide angles
# # set MEMBRANE_NORMAL {1 0 0}
# set METAL_NAMES {None}
# set CUTOFF_ANG 25
# set OUTPUT_CSV [tk_getSaveFile -initialdir "data" -initialfile "results.csv" -filetypes {{"CSV files" ".csv"} {"All files" "*"}}]
# source tcl_scripts/phi_theta_analysis.tcl

# axes location Off

# set _renderfile [tk_getSaveFile -initialdir "." -initialfile "15.tga" -defaultextension ".tga" -filetypes {{"All files" "*"}}]
# if {$_renderfile eq ""} { error "No render file selected, aborting." }

# render TachyonInternal $_renderfile "explorer /select,%s"
