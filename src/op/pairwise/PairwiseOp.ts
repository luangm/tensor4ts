import Tensor from "../../Tensor";
import ShapeUtils from "../../utils/ShapeUtils";
import Operation from "../Operation";

/**
 * Pairwise Op supports Broadcasting.
 */
export default abstract class PairwiseOp extends Operation {

  private readonly _x: Tensor;
  private readonly _y: Tensor;
  private readonly _z: Tensor;

  get x() {
    return this._x;
  }

  get y() {
    return this._y;
  }

  get z() {
    return this._z;
  }

  protected constructor(x: Tensor, y: Tensor, z: Tensor) {
    super([x, y], [z]);
    this._z = z;
    let xShape = ShapeUtils.getBroadcastedShape(x.shape, z.shape);
    let yShape = ShapeUtils.getBroadcastedShape(y.shape, z.shape);
    this._x = x.reshape(xShape);
    this._y = y.reshape(yShape);
  }

  abstract body(a: number, b: number): number;

  exec(dim?: number | number[]): void {
    // nothing
  }

}