import PairwiseOp from "../op/pairwise/PairwiseOp";
import ShapeUtils from "../utils/ShapeUtils";

export class PairwiseExecutor {

  public exec(op: PairwiseOp): void {
    switch (op.result.rank) {
      case 0:
        this.exec0Scalar(op);
        break;
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

  private exec0Scalar(op: PairwiseOp): void {
    let input = op.input.data;
    let other = op.other.data;
    let result = op.result.data;

    result[0] = op.body(input[0], other[0]);
  }

  private exec1Vector(op: PairwiseOp): void {
    let result = op.result.data;
    let shape = op.result.shape;

    let inputBroadShape = ShapeUtils.getBroadcastedShape(op.input.shape, shape);
    let otherBroadShape = ShapeUtils.getBroadcastedShape(op.other.shape, shape);

    let inputReshaped = op.input.reshape(inputBroadShape);
    let otherReshaped = op.other.reshape(otherBroadShape);

    let input = inputReshaped.data;
    let other = otherReshaped.data;

    let inputS0 = (inputBroadShape[0] === 1 ? 0 : inputReshaped.strides[0]) | 0;
    let otherS0 = (otherBroadShape[0] === 1 ? 0 : otherReshaped.strides[0]) | 0;
    let resultS0 = op.result.strides[0] | 0;
    let s0 = shape[0] | 0;

    let iPtr = 0;
    let oPtr = 0;
    let rPtr = 0;

    for (let i = 0; i < s0; i++) {
      result[rPtr] = op.body(input[iPtr], other[oPtr]);
      iPtr = (iPtr + inputS0) | 0;
      oPtr = (oPtr + otherS0) | 0;
      rPtr = (rPtr + resultS0) | 0;
    }
  }

  private exec2Matrix(op: PairwiseOp): void {

    let result = op.result.data;
    let shape = op.result.shape;

    let inputBroadShape = ShapeUtils.getBroadcastedShape(op.input.shape, shape);
    let otherBroadShape = ShapeUtils.getBroadcastedShape(op.other.shape, shape);

    let inputReshaped = op.input.reshape(inputBroadShape);
    let otherReshaped = op.other.reshape(otherBroadShape);

    let input = inputReshaped.data;
    let other = otherReshaped.data;

    let inputS0 = (inputBroadShape[0] === 1 ? 0 : inputReshaped.strides[0]) | 0;
    let inputS1 = (inputBroadShape[1] === 1 ? 0 : inputReshaped.strides[1]) | 0;
    let otherS0 = (otherBroadShape[0] === 1 ? 0 : otherReshaped.strides[0]) | 0;
    let otherS1 = (otherBroadShape[1] === 1 ? 0 : otherReshaped.strides[1]) | 0;
    let resultS0 = op.result.strides[0] | 0;
    let resultS1 = op.result.strides[1] | 0;
    let s0 = shape[0] | 0;
    let s1 = shape[1] | 0;

    let iPtr = 0;
    let oPtr = 0;
    let rPtr = 0;

    let inputD0 = (inputS0 - inputS1 * s1) | 0;
    let otherD0 = (otherS0 - otherS1 * s1) | 0;
    let resultD0 = (resultS0 - resultS1 * s1) | 0;

    let inputD1 = inputS1 | 0;
    let otherD1 = otherS1 | 0;
    let resultD1 = resultS1 | 0;

    for (let i = 0; i < s0; i++) {

      for (let j = 0; j < s1; j++) {
        result[rPtr] = op.body(input[iPtr], other[oPtr]);
        iPtr = (iPtr + inputD1) | 0;
        oPtr = (oPtr + otherD1) | 0;
        rPtr = (rPtr + resultD1) | 0;
      }

      iPtr = (iPtr + inputD0) | 0;
      oPtr = (oPtr + otherD0) | 0;
      rPtr = (rPtr + resultD0) | 0;
    }
  }

  private exec9General(op: PairwiseOp): void {
    let result = op.result.data;
    let shape = op.result.shape;

    let inputBroadShape = ShapeUtils.getBroadcastedShape(op.input.shape, shape);
    let otherBroadShape = ShapeUtils.getBroadcastedShape(op.other.shape, shape);

    let inputReshaped = op.input.reshape(inputBroadShape);
    let otherReshaped = op.other.reshape(otherBroadShape);

    let input = inputReshaped.data;
    let other = otherReshaped.data;

    let inputPointer = 0;
    let otherPointer = 0;
    let resultPointer = 0;

    let rank = shape.length | 0;

    let MEM = []; // [ RevSlots(rank), shape, is, os, rs, ...]
    let iS = new Int32Array(rank);
    let oS = new Int32Array(rank);
    let rS = new Int32Array(rank);

    for (let i = 0; i < rank; i++) {
      MEM.push(0);
    }
    for (let i = 0; i < rank; i++) {
      let r = rank - 1 - i;
      MEM.push(shape[r]);
      iS[i] = (inputBroadShape[r] === 1 ? 0 : inputReshaped.strides[r]) | 0;
      oS[i] = (otherBroadShape[r] === 1 ? 0 : otherReshaped.strides[r]) | 0;
      rS[i] = op.result.strides[r] | 0;
      MEM.push(iS[i] - (i > 0 ? iS[i - 1] * shape[rank - i] : 0));
      MEM.push(oS[i] - (i > 0 ? oS[i - 1] * shape[rank - i] : 0));
      MEM.push(rS[i] - (i > 0 ? rS[i - 1] * shape[rank - i] : 0));
    }

    let index = 0;
    let ptr = 0;
    for (let i = 0; i < result.length; i++) {
      ptr = rank | 0;
      index = 0;
      MEM[0] = (MEM[0] + 1) | 0;

      result[resultPointer] = op.body(input[inputPointer], other[otherPointer]);
      inputPointer = (inputPointer + MEM[ptr + 1]) | 0;
      otherPointer = (otherPointer + MEM[ptr + 2]) | 0;
      resultPointer = (resultPointer + MEM[ptr + 3]) | 0;

      while (MEM[index] === MEM[ptr] && index < rank - 1) {
        MEM[index] = 0;
        index = (index + 1) | 0;
        MEM[index] = (MEM[index] + 1) | 0;
        ptr = (ptr + 4) | 0;
        inputPointer = (inputPointer + MEM[ptr + 1]) | 0;
        otherPointer = (otherPointer + MEM[ptr + 2]) | 0;
        resultPointer = (resultPointer + MEM[ptr + 3]) | 0;
      }
    }
  }

}

export default new PairwiseExecutor();