import ConditionalOp from "../op/ternary/ConditionalOp";

export class ConditionalExecutor {

  public exec(op: ConditionalOp): void {
    switch (op.result.rank) {
      case 0:
      case 1:
        this.exec1Vector(op);
        break;
      case 2:
        this.exec2Matrix(op);
        break;
      default:
        throw Error();
    }
  }

  private exec1Vector(op: ConditionalOp): void {
    let cond = op.condition.data;
    let input = op.truthy.data;
    let other = op.falsy.data;
    let result = op.result.data;

    for (let i = 0; i < result.length; i++) {
      result[i] = op.body(cond[i], input[i], other[i]);
    }
  }

  private exec2Matrix(op: ConditionalOp): void {
    let cond = op.condition.data;
    let input = op.truthy.data;
    let other = op.falsy.data;
    let result = op.result.data;

    let condStrides = op.condition.strides;
    let inputStrides = op.truthy.strides;
    let otherStrides = op.falsy.strides;
    let resultStrides = op.result.strides;

    let shape = op.result.shape;

    for (let i = 0; i < shape[0]; i++) {
      for (let j = 0; j < shape[1]; j++) {
        let condPointer = i * condStrides[0] + j * condStrides[1];
        let inputPointer = i * inputStrides[0] + j * inputStrides[1];
        let otherPointer = i * otherStrides[0] + j * otherStrides[1];
        let resultPointer = i * resultStrides[0] + j * resultStrides[1];

        result[resultPointer] = op.body(cond[condPointer], input[inputPointer], other[otherPointer]);
      }
    }
  }

}

export default new ConditionalExecutor();