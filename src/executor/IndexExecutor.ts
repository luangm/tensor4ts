import IndexOp from "../op/index/IndexOp";

/**
 * Executor class is used to execute Ops
 * The executor implementation may be changed to use multiple threads / workers
 * An parallel optimization for execution could be split the inputs into multiple sub tensors and let worker run on each.
 */
export class IndexExecutor {

  exec(op: IndexOp): void {
    switch (op.result.rank) {
      case 0:
      case 1:
        this.exec1Vector(op);
        break;
    }
  }

  private exec1Vector(op: IndexOp): void {
    let input = op.input.data;
    let result = op.result.data;

    let accum = op.body(input[0]);
    for (let i = 0; i < input.length; i++) {
      let value = op.body(input[i]);
      let updated = op.update(accum, value, result[0], i);
      accum = updated[0];
      result[0] = updated[1];
    }
  }

}

export default new IndexExecutor();