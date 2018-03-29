import TransformOp from "./TransformOp";

export default class SoftplusOp extends TransformOp {

  body(a: number, b?: number): number {
    return Math.log1p(Math.exp(a));
  }

}