import subprocess
import re
import datetime

def gitinfo():
    git_ret=subprocess.Popen(['git','log','--pretty=%H','HEAD^..HEAD'],
        stdout=subprocess.PIPE)
    git_hash = git_ret.communicate()[0]
    if git_hash:
        git_hash=git_hash.strip().decode()
        url_ret=subprocess.Popen(['git','remote','show','origin'],
            stdout=subprocess.PIPE)
        remote=url_ret.communicate()[0].decode()
        match=re.search('URL:\s*(\S+)\n',remote)
        if match:
            git_url=match.group(1)
            scmversion='{0}:{1}'.format(git_url, git_hash)
        else:
            scmversion=git_hash
        return scmversion
    else:
        return None

def makefile():
    return open("Makefile").read()


def when():
    return datetime.datetime.now().isoformat()

if __name__=='__main__':
    attrib=dict()
    attrib["VERSION"]=gitinfo()
    attrib["CFG"]=makefile()
    attrib["COMPILETIME"]=when()

    f=open("sirdemo_version.hpp", "w")
    for k, v in attrib.items():
        f.write("constexpr char {0}[]=R\"({1})\";\n\n".format(k, v))
    f.close()

