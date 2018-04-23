import {DataType} from "../DataType";
import Shape from "../Shape";

export default class OldTensor {

  private readonly _data: Float32Array;
  private readonly _shape: Shape;

  get data() {
    return this._data;
  }

  get dataType() {
    return DataType.Float32;
  }

  get length() {
    return this._shape.length;
  }

  get rank() {
    return this._shape.rank;
  }

  get shape() {
    return this._shape.shape;
  }

  constructor(data: Float32Array, shape: Shape) {
    this._data = data;
    this._shape = shape;
  }

  //
  // abs(): Tensor {
  //   return TensorMath.abs(this);
  // }
  //
  // add(other: Tensor): Tensor {
  //   return TensorMath.add(this, other);
  // }
  //
  // broadcast(shape: number[]): Tensor {
  //   return TensorUtils.broadcastTensor(this, shape);
  // }
  //
  // ceil(): Tensor {
  //   return TensorMath.ceil(this);
  // }
  //
  // divide(other: Tensor): Tensor {
  //   return TensorMath.divide(this, other);
  // }
  //
  // dup(): Tensor {
  //   return TensorMath.dup(this);
  // }
  //
  // equal(other: Tensor): Tensor {
  //   return TensorMath.equal(this, other);
  // }
  //
  // erf(): Tensor {
  //   return TensorMath.erf(this);
  // }
  //
  // erfc(): Tensor {
  //   return TensorMath.erfc(this);
  // }
  //
  // exp(): Tensor {
  //   return TensorMath.exp(this);
  // }
  //
  // fill(scalar: number): Tensor {
  //   return TensorMath.fill(this, scalar);
  // }
  //
  // floor(): Tensor {
  //   return TensorMath.floor(this);
  // }
  //
  // floorDiv(other: Tensor): Tensor {
  //   return TensorMath.floorDiv(this, other);
  // }
  //
  // floorMod(other: Tensor): Tensor {
  //   return TensorMath.floorMod(this, other);
  // }
  //
  // get(indices: number | number[]): number {
  //   if (this.isZeros) {
  //     return 0;
  //   }
  //   if (!Array.isArray(indices)) {
  //     indices = this._shape.getIndices(indices);
  //   }
  //   let offset = this._shape.getOffset(indices) + this.offset;
  //   return this._data[offset];
  // }
  //
  // greater(other: Tensor): Tensor {
  //   return TensorMath.greater(this, other);
  // }
  //
  // greaterEqual(other: Tensor): Tensor {
  //   return TensorMath.greaterEqual(this, other);
  // }
  //
  // inspect() {
  //   return this.toString();
  // }
  //
  // less(other: Tensor): Tensor {
  //   return TensorMath.less(this, other);
  // }
  //
  // lessEqual(other: Tensor): Tensor {
  //   return TensorMath.lessEqual(this, other);
  // }
  //
  // log(): Tensor {
  //   return TensorMath.log(this);
  // }
  //
  // matmul(other: Tensor, transposeLeft = false, transposeRight = false): Tensor {
  //   return TensorMath.matmul(this, other, transposeLeft, transposeRight);
  // }
  //
  // multiply(other: Tensor): Tensor {
  //   return TensorMath.multiply(this, other);
  // }
  //
  // negate() {
  //   return TensorMath.negate(this);
  // }
  //
  // notEqual(other: Tensor): Tensor {
  //   return TensorMath.notEqual(this, other);
  // }
  //
  // reciprocal(): Tensor {
  //   return TensorMath.reciprocal(this);
  // }
  //
  // reciprocalGrad(): Tensor {
  //   return TensorMath.reciprocalGrad(this);
  // }
  //
  // reduceLogSumExp(dims: number | number[] = -1, keepDims = false): Tensor {
  //   return TensorMath.reduceLogSumExp(this, dims, keepDims);
  // }
  //
  // reduceMax(dims: number | number[] = -1, keepDims = false): Tensor {
  //   return TensorMath.reduceMax(this, dims, keepDims);
  // }
  //
  // reduceMean(dims: number | number[] = -1, keepDims = false): Tensor {
  //   return TensorMath.reduceMean(this, dims, keepDims);
  // }
  //
  // reduceMin(dims: number | number[] = -1, keepDims = false): Tensor {
  //   return TensorMath.reduceMin(this, dims, keepDims);
  // }
  //
  // reduceProd(dims: number | number[] = -1, keepDims = false): Tensor {
  //   return TensorMath.reduceProd(this, dims, keepDims);
  // }
  //
  // reduceSum(dims: number | number[] = -1, keepDims = false): Tensor {
  //   return TensorMath.reduceSum(this, dims, keepDims);
  // }
  //
  // repeat(multiple: number, dimension: number = -1): Tensor {
  //   return TensorMath.repeat(this, multiple, dimension);
  // }
  //
  // reshape(shape: number[]): Tensor {
  //   return TensorUtils.reshape(this, shape);
  // }
  //
  // round(): Tensor {
  //   return TensorMath.round(this);
  // }
  //
  // roundi(): Tensor {
  //   return TensorMath.round(this, this);
  // }
  //
  // set(indices: number | number[], value: number): void {
  //   if (!Array.isArray(indices)) {
  //     indices = this._shape.getIndices(indices);
  //   }
  //   let offset = this._shape.getOffset(indices) + this.offset;
  //   this._data[offset] = value;
  // }
  //
  // /**
  //  * Slice the tensor.
  //  *
  //  * if any begin is -ve, it means from the back.
  //  * If size is not specified or any size's dimension is -1. it means to the end.
  //  *
  //  * @param {number[]} begin - the indices to start the slice
  //  * @param {number[]} size - the size(shape) of the slice.
  //  * @returns {Tensor}
  //  */
  // slice(begin: number[], size?: number[]): Tensor {
  //   let offset = this.offset;
  //   let newShape = this.shape.slice();
  //   if (!size) {
  //     size = new Array(this.rank).fill(-1);
  //   }
  //
  //   for (let i = 0; i < this.rank; i++) {
  //     let a = begin[i] < 0 ? begin[i] + this.shape[i] : begin[i];
  //     offset += a * this.strides[i];
  //
  //     let width = size[i] < 0 ? (this.shape[i] - a) : Math.min(this.shape[i] - a, size[i]);
  //     newShape[i] = width;
  //   }
  //
  //   let shape = new Shape(newShape, this.strides, this._shape.order);
  //   return new Tensor(this._data, shape, offset);
  // }
  //
  // slice2(num: number): Tensor {
  //   let offset = this.offset;
  //   let newShape = [];
  //   let newStrides = [];
  //
  //   offset += num * this.strides[0];
  //
  //   for (let i = 1; i < this.rank; i++) {
  //     newShape.push(this.shape[i]);
  //     newStrides.push(this.strides[i]);
  //   }
  //
  //   let shape = new Shape(newShape, newStrides, this._shape.order);
  //   return new Tensor(this._data, shape, offset, this.isZeros);
  // }
  //
  // subtract(other: Tensor): Tensor {
  //   return TensorMath.subtract(this, other);
  // }
  //
  // tile(repeats: number[]): Tensor {
  //   return TensorMath.tile(this, repeats);
  // }
  //
  // toString() {
  //   return FloatTensor.tensorFormat.format(this);
  // }
  //
  // transpose(newAxis: number[] = []): Tensor {
  //   return TensorUtils.transpose(this, newAxis);
  // }
  //
  // truncDiv(other: Tensor): Tensor {
  //   return TensorMath.truncDiv(this, other);
  // }
  //
  // truncMod(other: Tensor): Tensor {
  //   return TensorMath.truncMod(this, other);
  // }

}