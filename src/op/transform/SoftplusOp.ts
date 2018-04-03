import TransformOp from "./TransformOp";
import Tensor from "../../Tensor";

export default class SoftplusOp extends TransformOp {

  constructor(base: Tensor, result: Tensor) {
    super(base, result);
  }

  body(a: number): number {
    return Math.log1p(Math.exp(a));
  }

}