import {DataType} from "../DataType";
import Tensor, {TensorBufferLike} from "../Tensor";
import TensorBase from "./TensorBase";

export default class FloatTensor extends TensorBase {

  private readonly _data: Float32Array;

  get data() {
    return this._data;
  }

  get dataType() {
    return DataType.Float32;
  }

  constructor(data: Float32Array, shape: number[], strides?: number[], offset: number = 0) {
    super(data, shape, strides, offset);
    this._data = data;
  }

  protected create(data: TensorBufferLike, shape: number[], strides?: number[], offset?: number): Tensor {
    return new FloatTensor(data as Float32Array, shape, strides, offset);
  }

  //
  // abs(): Tensor {
  //   return TensorMath.abs(this);
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

  //

  //
  // subtract(other: Tensor): Tensor {
  //   return TensorMath.subtract(this, other);
  // }
  //
  // tile(repeats: number[]): Tensor {
  //   return TensorMath.tile(this, repeats);
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