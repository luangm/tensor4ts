import TransformOp from "./TransformOp";
import Tensor from "../../Tensor";

export default class TanhGradOp extends TransformOp {

  constructor(base: Tensor, result: Tensor) {
    super(base, result);
  }

  body(a: number): number {
    let sech = 1 / Math.cosh(a);
    return sech * sech;
  }

}