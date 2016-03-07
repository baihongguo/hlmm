import os,argparse

parser=argparse.ArgumentParser()

parser.add_argument('qfile',type=str)

args=parser.parse_args()

qfile=file(args.qfile,'r')

qline=qfile.readline()

while len(qline)>0:
    qline=qline.split(' ')
    job_id=qline[0]
    os.system('qdel '+job_id)
    qline=qfile.readline()

qfile.close()