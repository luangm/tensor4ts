import TransformOp from "./TransformOp";

export default class Expm1Op extends TransformOp {

  body(a: number, b?: number): number {
    return Math.expm1(a);
  }

}