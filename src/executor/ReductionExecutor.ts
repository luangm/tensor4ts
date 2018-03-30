import ReductionOp from "../op/reduction/ReductionOp";

/**
 * Executor for Reduction/Accumulation Ops
 */
export class ReductionExecutor {

  exec(op: ReductionOp): void {
    switch (op.result.rank) {
      case 0:
      case 1:
        this.execReduce1Vector(op);
        break;
      case 2:
        this.execReduce2Matrix(op);
        break;
      default:
        this.execReduce9General(op);
        break;
    }
  }

  private execReduce1Vector(op: ReductionOp): void {
    let input = op.input.data;
    let result = op.result.data;
    if (op.initialValue !== 0) {
      op.result.filli(op.initialValue);
    }

    for (let i = 0; i < input.length; i++) {
      let value = op.body(input[i]);
      result[0] = op.update(result[0], value);
    }

    if (op.shouldPostProcess) {
      let n = input.length;
      result[0] = op.getResult(result[0], n);
    }
  }

  private execReduce2Matrix(op: ReductionOp): void {
    let reducedDims = op.reducedDims;
    let input = op.input.data;
    let result = op.result.data;
    if (op.initialValue !== 0) {
      op.result.filli(op.initialValue);
    }

    let inputStrides = op.input.strides;
    let resultStrides = op.result.strides;

    let shape = op.input.shape; // accumulate around input, not the result
    let s0 = shape[0];
    let s1 = shape[1];
    let is0 = inputStrides[0];
    let is1 = inputStrides[1];
    let rs0 = reducedDims[0] ? 0 : resultStrides[0];
    let rs1 = reducedDims[1] ? 0 : resultStrides[1];

    for (let i = 0; i < s0; i++) {
      for (let j = 0; j < s1; j++) {
        let inputPointer = i * is0 + j * is1;
        let resultPointer = i * rs0 + j * rs1;
        let value = op.body(input[inputPointer]);
        result[resultPointer] = op.update(result[resultPointer], value);
      }
    }

    if (op.shouldPostProcess) {
      let n = 1;
      for (let i = 0; i < reducedDims.length; i++) {
        if (reducedDims[i]) {
          n *= shape[i];
        }
      }

      for (let i = 0; i < result.length; i++) {
        result[i] = op.getResult(result[i], n);
      }
    }
  }

  private execReduce9General(op: ReductionOp): void {
    let reducedDims = op.reducedDims;
    let input = op.input.data;
    let result = op.result.data;
    if (op.initialValue !== 0) {
      op.result.filli(op.initialValue);
    }

    let shape = op.input.shape;
    let rank = shape.length | 0;

    let inputPointer = 0;
    let resultPointer = 0;

    let MEM = []; // [ RevSlots(rank), shape, is, rs, ...]
    let iS = new Array(rank).fill(0);
    let rS = new Array(rank).fill(0);

    for (let i = 0; i < rank; i++) {
      MEM.push(0);
    }
    for (let i = 0; i < rank; i++) {
      let r = rank - 1 - i;
      MEM.push(shape[r]);
      iS[i] = op.input.strides[r] | 0;
      rS[i] = (reducedDims[r] ? 0 : op.result.strides[r]) | 0;
      MEM.push(iS[i] - (i > 0 ? iS[i - 1] * shape[rank - i] : 0));
      MEM.push(rS[i] - (i > 0 ? rS[i - 1] * shape[rank - i] : 0));
    }

    let index = 0;
    let ptr = 0;
    for (let i = 0; i < input.length; i++) {
      ptr = rank | 0;
      index = 0;
      MEM[0] = (MEM[0] + 1) | 0;

      let value = op.body(input[inputPointer]);
      result[resultPointer] = op.update(result[resultPointer], value);
      inputPointer = (inputPointer + MEM[ptr + 1]) | 0;
      resultPointer = (resultPointer + MEM[ptr + 2]) | 0;

      while (MEM[index] === MEM[ptr] && index < rank - 1) {
        MEM[index] = 0;
        index = (index + 1) | 0;
        MEM[index] = (MEM[index] + 1) | 0;
        ptr = (ptr + 3) | 0;
        inputPointer = (inputPointer + MEM[ptr + 1]) | 0;
        resultPointer = (resultPointer + MEM[ptr + 2]) | 0;
      }
    }

    if (op.shouldPostProcess) {
      let n = 1;
      for (let i = 0; i < reducedDims.length; i++) {
        if (reducedDims[i]) {
          n *= shape[i];
        }
      }

      for (let i = 0; i < result.length; i++) {
        result[i] = op.getResult(result[i], n);
      }
    }
  }

}

export default new ReductionExecutor();