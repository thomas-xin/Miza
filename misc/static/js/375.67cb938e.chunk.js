(this.webpackJsonpmizaweb2=this.webpackJsonpmizaweb2||[]).push([[375,187],{267:function(e,n){function a(e){return e?"string"===typeof e?e:e.source:null}function s(e){return t("(?=",e,")")}function t(){for(var e=arguments.length,n=new Array(e),s=0;s<e;s++)n[s]=arguments[s];var t=n.map((function(e){return a(e)})).join("");return t}function i(e){var n=e[e.length-1];return"object"===typeof n&&n.constructor===Object?(e.splice(e.length-1,1),n):{}}function r(){for(var e=arguments.length,n=new Array(e),s=0;s<e;s++)n[s]=arguments[s];var t=i(n),r="("+(t.capture?"":"?:")+n.map((function(e){return a(e)})).join("|")+")";return r}e.exports=function(e){var n=t(/[A-Z_]/,t("(?:",/[A-Z0-9_.-]*:/,")?"),/[A-Z0-9_.-]*/),a={className:"symbol",begin:/&[a-z]+;|&#[0-9]+;|&#x[a-f0-9]+;/},i={begin:/\s/,contains:[{className:"keyword",begin:/#?[a-z_][a-z1-9_-]+/,illegal:/\n/}]},c=e.inherit(i,{begin:/\(/,end:/\)/}),l=e.inherit(e.APOS_STRING_MODE,{className:"string"}),g=e.inherit(e.QUOTE_STRING_MODE,{className:"string"}),o={endsWithParent:!0,illegal:/</,relevance:0,contains:[{className:"attr",begin:/[A-Za-z0-9._:-]+/,relevance:0},{begin:/=\s*/,relevance:0,contains:[{className:"string",endsParent:!0,variants:[{begin:/"/,end:/"/,contains:[a]},{begin:/'/,end:/'/,contains:[a]},{begin:/[^\s"'=<>`]+/}]}]}]};return{name:"HTML, XML",aliases:["html","xhtml","rss","atom","xjb","xsd","xsl","plist","wsf","svg"],case_insensitive:!0,contains:[{className:"meta",begin:/<![a-z]/,end:/>/,relevance:10,contains:[i,g,l,c,{begin:/\[/,end:/\]/,contains:[{className:"meta",begin:/<![a-z]/,end:/>/,contains:[i,c,g,l]}]}]},e.COMMENT(/<!--/,/-->/,{relevance:10}),{begin:/<!\[CDATA\[/,end:/\]\]>/,relevance:10},a,{className:"meta",begin:/<\?xml/,end:/\?>/,relevance:10},{className:"tag",begin:/<style(?=\s|>)/,end:/>/,keywords:{name:"style"},contains:[o],starts:{end:/<\/style>/,returnEnd:!0,subLanguage:["css","xml"]}},{className:"tag",begin:/<script(?=\s|>)/,end:/>/,keywords:{name:"script"},contains:[o],starts:{end:/<\/script>/,returnEnd:!0,subLanguage:["javascript","handlebars","xml"]}},{className:"tag",begin:/<>|<\/>/},{className:"tag",begin:t(/</,s(t(n,r(/\/>/,/>/,/\s/)))),end:/\/?>/,contains:[{className:"name",begin:n,relevance:0,starts:o}]},{className:"tag",begin:t(/<\//,s(t(n,/>/))),contains:[{className:"name",begin:n,relevance:0},{begin:/>/,relevance:0,endsParent:!0}]}]}}},762:function(e,n,a){!function e(){e.warned||(e.warned=!0,console.log('Deprecation (warning): Using file extension in specifier is deprecated, use "highlight.js/lib/languages/xml" instead of "highlight.js/lib/languages/xml.js"'))}(),e.exports=a(267)}}]);
//# sourceMappingURL=375.67cb938e.chunk.js.map