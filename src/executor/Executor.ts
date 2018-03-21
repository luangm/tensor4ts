import PairwiseOp from "../op/pairwise/PairwiseOp";
import ShapeUtils from "../utils/ShapeUtils";
import Operation from "../op/Operation";
import TransformOp from "../op/transform/TransformOp";
import ReductionOp from "../op/reduction/ReductionOp";

/**
 * Executor class is used to execute Ops
 * The executor implementation may be changed to use multiple threads / workers
 * An parallel optimization for execution could be split the inputs into multiple sub tensors and let worker run on each.
 */
export class Executor {

  /**
   * Runs an op. Does NOT return.
   * The caller is expected to grab result from op.result
   *
   * This function loops through the Tensor with consideration of buffer index
   */
  exec(op: Operation): void {

    if (op.isSpecial) {
      op.exec();
      return;
    }

    if (op instanceof PairwiseOp) {
      Executor._execPairwise(op);
      return;
    }

    if (op instanceof TransformOp) {
      Executor._execTransform(op);
      return;
    }

    if (op instanceof ReductionOp) {
      Executor._execReduce(op);
      return;
    }

    throw new Error("Cannot Execute Unknown Op");
  }

  private static _execPairwise(op: PairwiseOp): void {
    switch (op.result.rank) {
      case 0:
        Executor._execPairwiseScalar(op);
        break;
      case 1:
        Executor._execPairwiseVector(op);
        break;
      case 2:
        Executor._execPairwiseMatrix(op);
        break;
      default:
        Executor._execPairwiseGeneral(op);
        break;
    }
  }

  private static _execPairwiseGeneral(op: PairwiseOp): void {
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

  private static _execPairwiseMatrix(op: Operation): void {

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

  private static _execPairwiseScalar(op: Operation): void {
    let input = op.input.data;
    let other = op.other.data;
    let result = op.result.data;

    result[0] = op.body(input[0], other[0]);
  }

  private static _execPairwiseVector(op: Operation): void {
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

  private static _execReduce(op: ReductionOp): void {
    switch (op.result.rank) {
      case 0:
      case 1:
        this._execReduceVector(op);
        break;
      case 2:
        this._execReduceMatrix(op);
        break;
      default:
        this._execReduceGeneral(op);
        break;
    }
  }

  private static _execReduceGeneral(op: ReductionOp): void {
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

  private static _execReduceMatrix(op: ReductionOp): void {
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

  private static _execReduceVector(op: ReductionOp): void {
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

  private static _execTransform(op: TransformOp): void {
    switch (op.result.rank) {
      case 0:
      case 1:
        Executor._execTransformVector(op);
        break;
      case 2:
        Executor._execTransformMatrix(op);
        break;
      default:
        Executor._execTransformGeneral(op);
        break;
    }
  }

  private static _execTransformGeneral(op: TransformOp): void {
    let input = op.input.data;
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
      iS[i] = op.input.strides[r] | 0;
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

  private static _execTransformMatrix(op: TransformOp): void {
    let input = op.input.data;
    let result = op.result.data;

    let inputStrides = op.input.strides;
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

  private static _execTransformVector(op: TransformOp): void {
    let input = op.input.data;
    let result = op.result.data;

    for (let i = 0; i < result.length; i++) {
      result[i] = op.body(input[i]);
    }
  }
}

export default new Executor();