import sys,os,json,math

def flaten_script(script):
  string = "# " + "0\n"
  for line in script:
    string += str(line[0]);
    string += " " + str(line[1][0]);
    string += " " + str(line[1][1] + 0.15);
    string += " " + str(line[1][2] + 0.1);
    string += " " + str(line[2][0]);
    string += " " + str(line[2][1]); 
    string += " " + str(line[2][2]); 
    string += '\n'
  return string

def write_string_to_file(string, fname):
  with open(fname,'w') as sfile:
    sfile.write(string)
    print "writhing to %s : %d chars" % (fname,len(string))
  return

def load_json(filename):
  if filename[-5:] == '.json':
    return json.load(open(filename))
  else:
    #print filename + " is not a json file, skipped"
    return None

def load_task_file_under_path(path):
  multi_scripts = []
  for i,filename in enumerate(sorted(os.listdir(path))):
    full_path = path+filename
    scri = load_json(full_path)
    if scri is None:
      continue
    scri = flaten_script(scri)
    write_string_to_file(scri, full_path[:-5]+".traj")

    multi_scripts.append(scri)
    #print multi_scripts
  return multi_scripts


#path = "/home/haoxuw/mcgill/kinova/src/kinova-ros/script/tracked_results/"
path = "/home/haoxuw/mcgill/kinova/src/kinova-ros/script/tracked_pick_place/"

load_task_file_under_path(path);
