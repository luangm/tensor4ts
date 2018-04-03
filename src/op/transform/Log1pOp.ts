import TransformOp from "./TransformOp";
import Tensor from "../../Tensor";

export default class Log1pOp extends TransformOp {

  constructor(base: Tensor, result: Tensor) {
    super(base, result);
  }

  body(a: number): number {
    return Math.log1p(a);
  }

}