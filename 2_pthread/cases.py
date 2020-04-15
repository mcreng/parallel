import os
import subprocess
import re

inputs = [
    './datasets/sample.in',
    './datasets/1k.in',
    './datasets/20k.in',
    './datasets/20k2k.in',
    './datasets/input1.txt',
    './datasets/input2.txt',
    './datasets/input3.txt'
]

threads = [
    '1', '2', '4', '8', '16'
]

out = {k: {t: {'score': None, 'time': None, 'wrong': False}
           for t in threads + ['serial']} for k in inputs}


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
        'g++ -std=c++11 -pthread main.cpp pthreads_smith_waterman.cpp -o pthreads_smith_waterman')

    for inp in inputs:
        _, stdout, stderr = run(['./serial/serial_smith_waterman', inp])

        on_std_out(stdout.decode('utf-8'),
                   './serial/serial_smith_waterman', inp, 'serial')
        on_std_err(stderr.decode('utf-8'),
                   './serial/serial_smith_waterman', inp, 'serial')

        for thread in threads:
            print(thread, inp)
            _, stdout, stderr = run(['./pthreads_smith_waterman', inp, thread])
            on_std_out(stdout.decode('utf-8'),
                       './pthreads_smith_waterman', inp, thread)
            on_std_err(stderr.decode('utf-8'),
                       './pthreads_smith_waterman', inp, thread)
    printWhenDone()


def on_std_out(stdout, cmd, inp, num):
    stdout = re.sub(r'\s+', '', stdout)
    if re.match(r'\d+', stdout):
        score = re.findall(r'\d+', stdout)[0]
        out[inp][num]['score'] = score
    else:
        out[inp][num]['score'] = ''
        print(cmd, inp, num, stdout)


def on_std_err(stderr, cmd, inp, num):
    stderr = re.sub(r'\s+', '', stderr)
    if re.match(r'^Time:([\d.e-]+)s$', stderr):
        time = re.findall(r'^Time:([\d.e-]+)s$', stderr)[0]
        out[inp][num]['time'] = time
    else:
        out[inp][num]['time'] = ''
        print(cmd, inp, num, stderr)


def printWhenDone():
    wrong = []
    for inp in inputs:
        for thread in threads:
            if out[inp][thread]['score'] != out[inp]['serial']['score']:
                print(inp, thread, out[inp][thread]
                      ['score'], out[inp]['serial']['score'])
                out[inp][thread]['wrong'] = True
                wrong.append((inp, thread))
    f = chr(10).join(map(
        lambda thread: f"| n = {thread} | {' | '.join(map(lambda i: out[i][thread]['time'], inputs))} | ", threads))
    md = f"\
    # Time \n\
           \n\
    |        | {' | '.join(map(lambda i: '`' + i + '`', inputs))} | \n\
    | ------ | {' | '.join(map(lambda i: ' ----- ', inputs))} | \n\
    | serial | {' | '.join(map(lambda i: out[i]['serial']['time'], inputs))} | \n\
    {f}    "

    print(md)
    with open('time.md', 'w') as f:
        f.writelines(md)


if __name__ == '__main__':
    main()
