import os
import subprocess
import re

inputs = [
    './datasets/sample.in',
    './datasets/1k.in',
    './datasets/4k.in',
    './datasets/20k.in',
    './datasets/20k2k.in',
    './datasets/input1.txt',
    './datasets/input2.txt',
    './datasets/input3.txt',
    './datasets/input4.txt',
    './datasets/input5.txt',
    './datasets/input6.txt',
]

settings = [
    "{},{}".format(x, y) for x in [4, 8, 16, 32] for y in [32, 64, 128, 256, 512, 1024]
]

out = {k: {t: {'score': None, 'time': None, 'wrong': False}
           for t in settings + ['serial']} for k in inputs}


def run(cmd):
    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            cwd=os.getcwd()
                            )
    stdout, stderr = proc.communicate()

    return proc.returncode, stdout, stderr


def main():
    print('compile')
    os.system(
        '''/usr/local/cuda/bin/nvcc -std=c++11  -arch=compute_52 -code=sm_52  main.cu    cuda_smith_waterman.cu    -o cuda_smith_waterman    '''
    )

    for inp in inputs:
        print(inp, 'serial')
        _, stdout, stderr = run(['./serial/serial_smith_waterman', inp])

        on_std_out(stdout.decode('utf-8'),
                   './serial/serial_smith_waterman', inp, 'serial')
        on_std_err(stderr.decode('utf-8'),
                   './serial/serial_smith_waterman', inp, 'serial')

        for setting in settings:
            [block, thread] = setting.split(',')
            print(inp, block, thread)
            _, stdout, stderr = run(['./cuda_smith_waterman', inp, block, thread])
            on_std_out(stdout.decode('utf-8'),
                       './cuda_smith_waterman', inp, setting)
            on_std_err(stderr.decode('utf-8'),
                       './cuda_smith_waterman', inp, setting)
    printWhenDone()


def on_std_out(stdout, cmd, inp, num):
    stdout = re.sub(r'\s+', '', stdout)
    if re.search(r'\d+', stdout):
        score = re.findall(r'\d+', stdout)[0]
        out[inp][num]['score'] = score
    else:
        out[inp][num]['score'] = ''
        print(cmd, inp, num, stdout)


def on_std_err(stderr, cmd, inp, num):
    stderr = re.sub(r'\s+', '', stderr)
    if re.search(r'Time:([\d.e-]+)s', stderr):
        times = re.findall(r'Time:([\d.e-]+)s', stderr)
        if len(times) == 1:
            out[inp][num]['time'] = times[0]
        else:
            out[inp][num]['time'] = "%s<br>%s"%(times[0],times[1])
    else:
        out[inp][num]['time'] = ''
        print(cmd, inp, num, stderr)


def printWhenDone():
    wrong = []
    for inp in inputs:
        for thread in settings:
            if out[inp][thread]['score'] != out[inp]['serial']['score']:
                print(inp, thread, out[inp][thread]
                ['score'], out[inp]['serial']['score'])
                out[inp][thread]['wrong'] = True
                wrong.append((inp, thread))
    f = chr(10).join(map(
        lambda thread: f"\
| n = {thread} | {' | '.join(map(lambda i: out[i][thread]['time'], inputs))} | ", settings))
    md = f"\
# Time \n\
       \n\
|        | {' | '.join(map(lambda i: '`' + i + '`', inputs))} | \n\
| ------ | {' | '.join(map(lambda i: ' ----- ', inputs))} | \n\
| serial | {' | '.join(map(lambda i: out[i]['serial']['time'], inputs))} | \n\
{f}"

    print(md)
    with open('time.md', 'w') as f:
        f.writelines(md)


if __name__ == '__main__':
    main()
