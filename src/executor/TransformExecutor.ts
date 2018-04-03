import TransformOp from "../op/transform/TransformOp";

export class TransformExecutor {

  exec(op: TransformOp): void {
    switch (op.result.rank) {
      case 0:
      case 1:
        this.exec1Vector(op);
        break;
      case 2:
        this.exec2Matrix(op);
        break;
      default:
        this.exec9General(op);
        break;
    }
  }

  private exec1Vector(op: TransformOp): void {
    let input = op.base.data;
    let result = op.result.data;

    for (let i = 0; i < result.length; i++) {
      result[i] = op.body(input[i]);
    }
  }

  private exec2Matrix(op: TransformOp): void {
    let input = op.base.data;
    let result = op.result.data;

    let inputStrides = op.base.strides;
    let resultStrides = op.result.strides;

    let shape = op.result.shape;

    for (let i = 0; i < shape[0]; i++) {
      for (let j = 0; j < shape[1]; j++) {
        let inputPointer = i * inputStrides[0] + j * inputStrides[1];
        let resultPointer = i * resultStrides[0] + j * resultStrides[1];

        result[resultPointer] = op.body(input[inputPointer]);
      }
    }
  }

  private exec9General(op: TransformOp): void {
    let input = op.base.data;
    let result = op.result.data;
    let shape = op.result.shape;
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
      iS[i] = op.base.strides[r] | 0;
      rS[i] = op.result.strides[r] | 0;
      MEM.push(iS[i] - (i > 0 ? iS[i - 1] * shape[rank - i] : 0));
      MEM.push(rS[i] - (i > 0 ? rS[i - 1] * shape[rank - i] : 0));
    }

    let index = 0;
    let ptr = 0;
    for (let i = 0; i < result.length; i++) {
      ptr = rank | 0;
      index = 0;
      MEM[0] = (MEM[0] + 1) | 0;

      result[resultPointer] = op.body(input[inputPointer]);
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
  }

}

export default new TransformExecutor();