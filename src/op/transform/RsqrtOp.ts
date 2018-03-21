import TransformOp from "./TransformOp";

export default class RsqrtOp extends TransformOp {

  body(a: number, b?: number): number {
    return 1 / Math.sqrt(a);
  }

}