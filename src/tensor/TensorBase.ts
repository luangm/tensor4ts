import {DataType} from "../DataType";
import Tensor from "../Tensor";
import TensorBuffer from "../TensorBuffer";
import TensorMath from "../TensorMath";
import ArrayUtils from "../utils/ArrayUtils";
import ShapeUtils from "../utils/ShapeUtils";
import TensorFormat from "../utils/TensorFormat";
import TensorFlags from "./TensorFlags";

export default abstract class TensorBase implements Tensor {

  private static FORMAT: TensorFormat = new TensorFormat({});

  abstract readonly data: TensorBuffer;
  abstract readonly dataType: DataType;

  private readonly _flags: TensorFlags;
  private readonly _length: number;
  private readonly _offset: number;
  private readonly _rank: number;
  private readonly _shape: number[];
  private readonly _strides: number[];

  get cContiguous() {
    return this._flags.cContiguous;
  }

  get fContiguous() {
    return this._flags.fContiguous;
  }

  get length() {
    return this._length;
  }

  get offset() {
    return this._offset;
  }

  get rank() {
    return this._rank;
  }

  get shape() {
    return this._shape;
  }

  get strides() {
    return this._strides;
  }

  protected constructor(data: TensorBuffer, shape: number[], strides?: number[], offset: number = 0) {
    this._shape = shape;
    this._rank = shape.length;
    this._strides = strides || ShapeUtils.getStrides(shape);
    this._length = ShapeUtils.getLength(shape);
    this._offset = offset;
    this._flags = ShapeUtils.inferFlags(this._shape, this._strides, this._offset);
  }

  add(other: Tensor): Tensor {
    return TensorMath.add(this, other);
  }

  broadcast(newShape: number[]): Tensor {
    let allowed = ShapeUtils.canBroadcastTo(this.shape, newShape);
    if (!allowed) {
      throw new Error("Cannot broadcast to " + newShape);
    }

    let newStrides = ArrayUtils.padLeft(this.strides, newShape.length - this.shape.length, 0);
    let broadcastIndices = ShapeUtils.getBroadcastIndices(this.shape, newShape);
    for (let index of broadcastIndices) {
      newStrides[index] = 0;
    }

    return this.create(this.data, newShape, newStrides, this.offset);
  }

  get(indices: number[]): number {
    let offset = this.offset;
    for (let i = 0; i < this.rank; i++) {
      offset += indices[i] * this.strides[i];
    }
    return this.data[offset];
  }

  inspect() {
    return this.toString();
  }

  // TODO: If not c-contiguous should make a copy
  reshape(shape: number[]): Tensor {
    let newShape = ShapeUtils.inferShape(this.length, shape);
    return this.create(this.data, newShape);
  }

  set(indices: number[], value: number): void {
    let offset = this.offset;
    for (let i = 0; i < this.rank; i++) {
      offset += indices[i] * this.strides[i];
    }
    this.data[offset] = value;
  }

  slice(begin: number[], size?: number[]): Tensor {
    let offset = this.offset;
    let newShape = this.shape.slice();
    if (!size) {
      size = new Array(this.rank).fill(-1);
    }

    for (let i = 0; i < this.rank; i++) {
      let a = begin[i] < 0 ? begin[i] + this.shape[i] : begin[i];
      offset += a * this.strides[i];
      newShape[i] = size[i] < 0 ? (this.shape[i] - a) : Math.min(this.shape[i] - a, size[i]);
    }

    return this.create(this.data, newShape, this.strides, offset);
  }

  sliceSingle(num: number): Tensor {
    let offset = this.offset + num * this.strides[0];
    let newShape = [];
    let newStrides = [];

    for (let i = 1; i < this.rank; i++) {
      newShape.push(this.shape[i]);
      newStrides.push(this.strides[i]);
    }

    return this.create(this.data, newShape, newStrides, offset);
  }

  toString() {
    return TensorBase.FORMAT.format(this);
  }

  protected abstract create(data: TensorBuffer, shape: number[], strides?: number[], offset?: number): Tensor;

}