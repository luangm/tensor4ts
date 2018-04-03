import TransformOp from "../TransformOp";
import Tensor from "../../../Tensor";

export default class SetOp extends TransformOp {

  private readonly _scalar: number;

  get scalar() {
    return this._scalar;
  }

  constructor(input: Tensor, result: Tensor, scalar: number) {
    super(input, result);
    this._scalar = scalar;
  }

  body(a: number): number {
    return this._scalar;
  }

}