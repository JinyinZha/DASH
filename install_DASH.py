import os

if not os.path.exists("DASH/config.py"):
    print("The script must be run in the DASH root folder")
    exit()

#Download and install ADFR
print("Installing ADFR...")
root = os.getcwd()
os.system("""mkdir adfr
cd adfr
wget -O installer.tar.gz https://ccsb.scripps.edu/adfr/download/1038/
tar -xzvf installer.tar.gz
cd ADFRsuite*
echo Y | ./install.sh -d ../adfr -c 0""")
adfr_path = "%s/adfr/adfr/bin"%(root)
f = open("DASH/config.py")
fli = f.read().split("\n")
f.close()
fli[0] = 'adfr_prepare = "%s/prepare_receptor"'%(adfr_path)
fli[1] = 'adfr_autosite = "%s/autosite"'%(adfr_path)
f = open("DASH/config.py","w")
f.write("\n".join(fli))
f.close()

#Create bin
print("Create bin...")
if os.path.exists("bin"):
    os.system("rm -rf bin")
os.system("""mkdir bin
cd bin
ln -s ../DASH/guide_sample_main.py DASH
ln -s ../DASH/cv.py DASH_Train
ln -s ../DASH/score_net.py Score_Net""")

print("Over!")

