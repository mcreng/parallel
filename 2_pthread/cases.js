const { execSync, spawnSync } = require('child_process')
const fs = require('fs')

const inputs = [
  'sample.in',
  '1k.in',
  '20k.in',
  '20k2k.in',
  'input1.txt',
  'input2.txt',
  'input3.txt'
].map(input => './datasets/' + input)

const threads = [
  '1', '2', '4', '8', '16'
]

const out = {}

const main = () => {
  inputs.forEach(input => {
    const empty = { score: null, time: null }
    out[input] = { serial: { ...empty } }
    threads.forEach(thread => {
      out[input][thread] = { ...empty }
    })
  })

  console.log('compile')
  execSync('g++ -std=c++11 -pthread main.cpp pthreads_smith_waterman_skeleton.cpp -o pthreads_smith_waterman')

  inputs.forEach(input => {
    console.log('serial', input)
    const { stdout, stderr } = spawnSync('./serial/serial_smith_waterman', [input],)
    onStdOut(stdout.toString(), './serial/serial_smith_waterman', input, 'serial')
    onStdErr(stderr.toString(), './serial/serial_smith_waterman', input, 'serial')
    threads.forEach(thread => {
      console.log(thread, input)
      const { stdout, stderr } = spawnSync('./pthreads_smith_waterman', [input, thread],)
      onStdOut(stdout.toString(), './pthreads_smith_waterman', input, thread)
      onStdErr(stderr.toString(), './pthreads_smith_waterman', input, thread)
    })
  })
  printWhenDone()
}

const onStdOut = (str, cmd, input, num) => {
  const pstr = str.replace(/\s+/g, '')
  if (/^(\d+)$/.test(pstr)) {
    const score = Number.parseInt(pstr.match(/\d+/)[0])
    out[input][num].score = score
  } else {
    out[input][num].score = ''
    console.log(cmd, input, num, str)
  }
}

const onStdErr = (str, cmd, input, num) => {
  const pstr = str.replace(/\s+/g, '')
  if (/^Time:([\d.e-]+)s$/.test(pstr)) {
    const time = Number.parseFloat(pstr.match(/^Time:([\d.e-]+)s$/)[1])
    out[input][num].time = time
  } else {
    out[input][num].time = ''
    console.log(cmd, input, num, str)
  }
}

const deepExists = (object, keyword) => {
  if (object === null) {
    return keyword === null
  } else if (typeof object === 'object') {
    return Object.keys(object)
      .reduce(
        (flag, key) => {
          if (flag) {
            return flag
          } else {
            return deepExists(object[key], keyword)
          }
        }, false
      )
  } else {
    return `${object}`.toLowerCase().includes(keyword)
  }
}

const printWhenDone = () => {
  if (deepExists(out, null)) return
  // console.log(out)
  const wrong = []
  inputs.forEach(input => {
    threads.forEach(thread => {
      if (out[input][thread].score !== out[input].serial.score) {
        console.log(input, thread, out[input][thread].score, out[input].serial.score)
        out[input][thread].wrong = true
        wrong.push({ input, thread })
      }
    })
  })
  const md = `
  # Time
  
  |        | ${inputs.map(i => '`' + i + '`').join(' | ')} |
  | ------ | ${inputs.map(i => ' ----- ').join(' | ')} |
  | serial | ${inputs.map(input => out[input].serial.time).join(' | ')} |
  ${threads.map(thread => `\
  | n=${thread} | ${inputs.map(input => out[input][thread].time).join(' | ')} |\
  `).join('\n')}
  `
  console.log(md)
  fs.writeFile('time.md', md, err => {
    if (err === null) {
      console.log('successully write to time.md')
    }
  })
  if (wrong.length > 0) {
    console.log(wrong)
  }
}

process.on('exit', printWhenDone.bind(null, { out }))

main()
