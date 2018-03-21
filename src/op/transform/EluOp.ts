import TransformOp from "./TransformOp";

export default class EluOp extends TransformOp {

  body(a: number, b?: number): number {
    return a > 0 ? a : Math.expm1(a);
  }

}