module.exports = async function* (events) {
  const records = [];
  for await (const event of events) {
    if (event.type === 'test:pass' || event.type === 'test:fail') {
      const code = event.data.details?.error?.cause?.code || event.data.details?.error?.code;
      records.push({id:event.data.name, status:event.type === 'test:pass' ? 'passed' : 'failed', failure_kind:code === 'ERR_ASSERTION' ? 'assertion' : event.type === 'test:fail' ? 'runtime' : null, error_code:code});
    }
    if (event.type === 'test:stdout' || event.type === 'test:stderr') {
      yield 'TEST_OUTPUT:' + JSON.stringify(event.data.message) + '\n';
    }
    if (event.type === 'test:fail') yield 'FAILURE:' + JSON.stringify(event.data) + '\n';
  }
  yield 'LOCALBENCH_TESTS_JSON:' + JSON.stringify({tests_run:records.length,records,successful:records.length>0 && records.every(r=>r.status==='passed')}) + '\n';
};
