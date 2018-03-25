import TransformOp from "../TransformOp";
import Tensor from "../../../Tensor";

export default class SetOp extends TransformOp {

  private _scalar: number;

  constructor(input: Tensor, other: Tensor, result: Tensor, scalar: number) {
    super(input, other, result);
    this._scalar = scalar;
  }

  body(a: number, b?: number): number {
    return this._scalar;
  }

}