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
    let truthy = op.truthy.data;
    let falsy = op.falsy.data;
    let result = op.result.data;
    let condZeros = op.condition.isZeros;
    let truthyZeros = op.truthy.isZeros;
    let falsyZeros = op.falsy.isZeros;

    for (let i = 0; i < result.length; i++) {
      let c = condZeros ? 0 : cond[i];
      let a = truthyZeros ? 0 : truthy[i];
      let b = falsyZeros ? 0 : falsy[i];
      result[i] = op.body(c, a, b);
    }
  }

  private exec2Matrix(op: ConditionalOp): void {
    let cond = op.condition.data;
    let truthy = op.truthy.data;
    let falsy = op.falsy.data;
    let result = op.result.data;

    let condStrides = op.condition.strides;
    let inputStrides = op.truthy.strides;
    let otherStrides = op.falsy.strides;
    let resultStrides = op.result.strides;

    let condZeros = op.condition.isZeros;
    let truthyZeros = op.truthy.isZeros;
    let falsyZeros = op.falsy.isZeros;

    let shape = op.result.shape;

    for (let i = 0; i < shape[0]; i++) {
      for (let j = 0; j < shape[1]; j++) {
        let condPointer = i * condStrides[0] + j * condStrides[1];
        let inputPointer = i * inputStrides[0] + j * inputStrides[1];
        let otherPointer = i * otherStrides[0] + j * otherStrides[1];
        let resultPointer = i * resultStrides[0] + j * resultStrides[1];

        let c = condZeros ? 0 : cond[condPointer];
        let a = truthyZeros ? 0 : truthy[inputPointer];
        let b = falsyZeros ? 0 : falsy[otherPointer];
        
        result[resultPointer] = op.body(c, a, b);
      }
    }
  }

}

export default new ConditionalExecutor();